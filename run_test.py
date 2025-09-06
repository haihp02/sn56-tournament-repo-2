from ast import Dict
from scripts import train_cst as cst
import logging
import sys
import docker
import asyncio
import uuid
import json 
import typer
from docker.models.containers import Container
# Set up logger to log to both file and console
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def build_wandb_env(task_id: str, hotkey: str) -> dict:
    wandb_path = f"{cst.WANDB_LOGS_DIR}/{task_id}_{hotkey}"

    env = {
        "WANDB_MODE": "offline",
        **{key: wandb_path for key in cst.WANDB_DIRECTORIES}
    }

    return env


# Create formatter
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Create file handler
file_handler = logging.FileHandler('run_test.log')
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(formatter)

# Create console handler
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(formatter)

# Add handlers to logger
logger.addHandler(file_handler)
logger.addHandler(console_handler)


async def create_volumes_if_dont_exist():
    client: docker.DockerClient = docker.from_env()
    volume_names = cst.VOLUME_NAMES
    for volume_name in volume_names:
        try:
            volume = client.volumes.get(volume_name)
        except docker.errors.NotFound:
            volume = client.volumes.create(name=volume_name)
            logger.info(f"Volume '{volume_name}' created.")


def stream_container_logs(container: Container):
    buffer = ""
    try:
        for log_chunk in container.logs(stream=True, follow=True):
            log_text = log_chunk.decode("utf-8", errors="replace")
            buffer += log_text
            while "\n" in buffer:
                line, buffer = buffer.split("\n", 1)
                if line:
                    logger.info(line)
        if buffer:
            logger.info(buffer)
    except Exception as e:
        logger.error(f"Error streaming logs: {str(e)}")


def build_docker_image(
    dockerfile_path: str, context_path: str = ".", is_image_task: bool = False, tag: str = None, no_cache: bool = True
) -> tuple[str, Exception | None]:
    client: docker.DockerClient = docker.from_env()

    if tag is None:
        tag =  f"standalone-text-trainer:test"

    logger.info(f"Building Docker image '{tag}', Dockerfile path: {dockerfile_path}, Context Path: {context_path}...")

    try:
        build_output = client.api.build(
            path=context_path,
            dockerfile=dockerfile_path,
            tag=tag,
            nocache=no_cache,
            decode=True,
        )
        
        # Consume the build output generator to actually trigger the build
        for chunk in build_output:
            if 'stream' in chunk:
                logger.info(chunk['stream'].strip())
            elif 'error' in chunk:
                error_msg = chunk['error']
                logger.error(f"Build error: {error_msg}")
                return tag, Exception(error_msg)
        
        logger.info(f"Docker image '{tag}' built successfully")
        return tag, None
        
    except Exception as e:
        logger.error(f"Failed to build Docker image: {str(e)}")
        return tag, e


def calculate_container_resources(gpu_ids: list[int]) -> tuple[str, int]:
    """Calculate memory limit and CPU limit based on GPU count.
    
    Returns:
        tuple: (memory_limit_str, cpu_limit_nanocpus)
    """
    num_gpus = len(gpu_ids)
    memory_limit = f"{num_gpus * cst.MEMORY_PER_GPU_GB}g"
    cpu_limit_nanocpus = num_gpus * cst.CPUS_PER_GPU * 1_000_000_000
    
    logger.info(f"Allocating resources for {num_gpus} GPUs: {memory_limit} memory, {num_gpus * cst.CPUS_PER_GPU} CPUs")
    return memory_limit, cpu_limit_nanocpus


async def run_trainer_container_text(
    task_id: str,
    hotkey: str,
    tag: str,
    model: str,
    dataset: str,
    dataset_type: dict,
    task_type: str,
    file_format: str,
    expected_repo_name: str,
    hours_to_complete: float,
    gpu_ids: list[int] = [0],
) -> Container:
    client: docker.DockerClient = docker.from_env()

    environment = build_wandb_env(task_id, hotkey)

    command: list[str] = [
        "--task-id",
        task_id,
        "--model",
        model,
        "--dataset",
        dataset,
        "--dataset-type",
        json.dumps(dataset_type),
        "--task-type",
        task_type,
        "--file-format",
        file_format,
        "--expected-repo-name",
        expected_repo_name,
        "--hours-to-complete",
        str(hours_to_complete)
    ]

    container_name = f"text-trainer-{uuid.uuid4().hex}"
    
    # Calculate resources based on GPU count
    memory_limit, cpu_limit_nanocpus = calculate_container_resources(gpu_ids)
    logger.info(f"MEMORY LIMIT: {memory_limit}")
    logger.info(f"CPU LIMIT: {cpu_limit_nanocpus}")
    
    try:
        container: Container = client.containers.run(
            image=tag,
            command=command,
            volumes={
                cst.VOLUME_NAMES[0]: {"bind": cst.OUTPUT_CHECKPOINTS_PATH, "mode": "rw"},
                cst.VOLUME_NAMES[1]: {"bind": cst.CACHE_ROOT_PATH, "mode": "rw"},
            },
            remove=False,
            name=container_name,
            mem_limit=memory_limit,
            nano_cpus=cpu_limit_nanocpus,
            shm_size="8g",
            device_requests=[docker.types.DeviceRequest(device_ids=[str(i) for i in gpu_ids], capabilities=[["gpu"]])],
            security_opt=["no-new-privileges"],
            cap_drop=["ALL"],
            detach=True,
            network_mode="none",
            environment=environment,
        )

        # log_streaming_task = asyncio.create_task(asyncio.to_thread(stream_container_logs, container, get_all_context_tags()))
        return container
    except Exception as e:
        logger.error(e)
        return e


def extract_container_error(logs: str) -> str | None:
    lines = logs.strip().splitlines()

    for line in reversed(lines):
        line = line.strip()
        if line and ":" in line and any(word in line for word in ["Error", "Exception"]):
            return line

    return None


def run_downloader_container(
    task_id: str,
    model: str,
    dataset_url: str,
    task_type: str,
    file_format: str | None = None,
) -> tuple[int, Exception | None]:
    client = docker.from_env()

    command = [
        "--task-id",
        task_id,
        "--model",
        model,
        "--task-type",
        task_type,
        "--dataset",
        dataset_url,
    ]
    if file_format:
        command += ["--file-format", file_format]

    container_name = f"downloader-{task_id}-{str(uuid.uuid4())[:8]}"
    container = None

    try:
        logger.info(f"Starting downloader container: {container_name}")
        container = client.containers.run(
            image=cst.TRAINER_DOWNLOADER_DOCKER_IMAGE,
            name=container_name,
            command=command,
            volumes={cst.VOLUME_NAMES[1]: {"bind": "/cache", "mode": "rw"}},
            remove=False,
            detach=True,
        )

        stream_container_logs(container)

        result = container.wait()
        exit_code = result.get("StatusCode", -1)

        if exit_code == 0:
            logger.info(f"Download completed successfully for task {task_id}")
        else:
            logs = container.logs().decode("utf-8", errors="ignore")
            error_message = extract_container_error(logs)
            return exit_code, error_message

        return exit_code, None

    except docker.errors.ContainerError as e:
        logger.error(f"Downloader container failed for task {task_id}: {e}")
        return 1, e

    except Exception as ex:
        logger.error(f"Unexpected error in downloader for task {task_id}: {ex}")
        return 1, ex

    finally:
        if container:
            try:
                container.remove(force=True)
            except Exception as cleanup_err:
                logger.warning(f"Failed to remove container {container_name}: {cleanup_err}")


async def run_trainer(task: dict, build_image: bool = False):
    timeout_seconds = int(task["hours_to_complete"] * 3600)
    task_type = task["task_type"]
    await create_volumes_if_dont_exist()

    local_repo_path = "."
    dockerfile_path = f"{local_repo_path}/dockerfiles/standalone-text-trainer.dockerfile"
    # logger.info("Running Cache Download Container")
    # await log_task(training_data.task_id, task.hotkey, "Downloading data")

    download_status, exc = await asyncio.to_thread(
        run_downloader_container,
        task_id=task["task_id"],
        model=task["model"],
        dataset_url=task["dataset"],
        task_type=task_type,
        file_format=task["file_format"],
    )
    
    if download_status != 0:
        logger.error(f"Download failed with status {download_status}: {exc}")
        return
    
    # now run the task in a container
    tag =  f"standalone-text-trainer:test"
    if build_image:
        print("Building image ...")
        tag, exc = await asyncio.to_thread(
                build_docker_image,
                dockerfile_path=dockerfile_path,
                is_image_task=False,
                context_path=local_repo_path,
                tag = tag
        )
    
    logger.info(f"Successfully built Docker image: {tag}")
    logger.info(f"Running task now")
    
    try:
        container = await asyncio.wait_for(
            run_trainer_container_text(
                task_id=task["task_id"],
                hotkey="ABC-TEST",
                tag=tag,
                model=task["model"],
                dataset=task["dataset"],
                dataset_type=task["dataset_type"],
                task_type=task_type,
                file_format=task["file_format"],
                expected_repo_name=task["expected_repo_name"],
                hours_to_complete=task["hours_to_complete"],
                gpu_ids=task.get("gpu_ids", [0]),
            ),
            timeout=60,
        )
        
        if isinstance(container, Exception):
            logger.error(f"Failed to start container: {container}")
            return
            
        # Stream logs and wait for container to complete
        logger.info("Starting to stream container logs...")
        await asyncio.to_thread(stream_container_logs, container)
        
        # Wait for container to complete with timeout
        wait_task = asyncio.create_task(asyncio.to_thread(container.wait))
        done, pending = await asyncio.wait({wait_task}, timeout=timeout_seconds)
       
        if wait_task in done:
            result = await wait_task
            logger.info(f"Container.wait() returned: {result}")
            status_code = result.get("StatusCode", -1)
            if status_code == 0:
                success = True
            else:
                success = False
        else:
            logger.error(f"Timeout reached ({timeout_seconds}s). Killing container...")
            # Cancel the wait task and stop the container
            wait_task.cancel()
            try:
                container.stop(timeout=10)  # Give container 10 seconds to stop gracefully
                logger.info("Container stopped due to timeout")
            except Exception as stop_err:
                logger.warning(f"Failed to stop container gracefully: {stop_err}")
                try:
                    container.kill()  # Force kill if stop fails
                    logger.info("Container force killed due to timeout")
                except Exception as kill_err:
                    logger.error(f"Failed to kill container: {kill_err}")
            success = False
            
    except asyncio.TimeoutError:
        logger.error("Timeout waiting for container to start")
    except Exception as e:
        logger.error(f"Error during training: {e}")
    finally:
        # Clean up container
        if 'container' in locals() and not isinstance(container, Exception):
            try:
                container.remove(force=True)
                logger.info("Container cleaned up")
            except Exception as cleanup_err:
                logger.warning(f"Failed to remove container: {cleanup_err}")


def main(task_path: str, build_image: bool = False):
    with open(task_path, "r") as f:
        task = json.load(f)
    asyncio.run(run_trainer(task, build_image))


if __name__ == "__main__":
   typer.run(main)
