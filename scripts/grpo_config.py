from model_utility import get_model_architecture, get_model_num_params, get_use_liger, disable_flash_attention, get_use_vllm, get_gradient_checkpointing, get_gpu_count
from copy import deepcopy


GRPO_CONFIG = {
    "0_1_b": {
        "lr": 0.00015,
        "distributed": "ddp",
        "gpu_count": 1,
        "batch_size": 40,
        "vllm_gpu_memory_utilization": 0.4
    },
    "1_2_b": {
        "lr": 7.5e-5,
        "distributed": "ddp",
        "gpu_count": 1,
        "batch_size": 40,
        "vllm_gpu_memory_utilization": 0.4
    },
    "2_4_b": {
        "lr": 8e-5,
        "distributed": "ddp",
        "gpu_count": 2,
        "batch_size": 42,
        "vllm_gpu_memory_utilization": 0.35,
        "use_lora": True
    },
    "4_5_b": {
        "lr": 8e-5,
        "distributed": "ddp",
        "gpu_count": 2,
        "batch_size": 42,
        "use_lora": True,
        "vllm_gpu_memory_utilization": 0.4
    },
    "5_6_b": {
        "lr": 6e-5,
        "distributed": "ddp",
        "gpu_count": 2,
        "batch_size": 42,
        "use_lora": True,
        "vllm_gpu_memory_utilization": 0.4
    },
    "6_9_b": {
        "lr": 6e-5,
        "distributed": "ddp",
        "gpu_count": 4,
        "batch_size": 24,
        "use_lora": True,
        "vllm_gpu_memory_utilization": 0.5
    },
    "9_12_b": {
        "lr": 6.5e-5,   
        "distributed": "ddp",
        "gpu_count": 4,
        "use_lora": True,
        "batch_size": 16,
        "vllm_gpu_memory_utilization": 0.6
    },
    "12_15_b": {
        "lr": 8e-5,
        "distributed": "ddp",
        "gpu_count": 4,
        "use_lora": True,
        "batch_size": 2,
        "vllm_gpu_memory_utilization": 0.8,
    },
    "15_20_b": {
        "lr": 8e-5,
        "distributed": "ddp",
        "gpu_count": 4,
        "use_lora": True,
        "batch_size": 16,
        "vllm_gpu_memory_utilization": 0.6,
        "use_vllm": False,
    },
    "20_40_b": {
        "lr": 8e-5,
        "distributed": "ddp",
        "gpu_count": 8,
        "use_lora": True,
        "batch_size": 16,
        "vllm_gpu_memory_utilization": 0.6,
        "use_vllm": False,
        "use_4bit": True
    },
    "40_80_b": {
        "lr": 8e-5,
        "distributed": "ddp",
        "gpu_count": 8,
        "use_lora": True,
        "batch_size": 2,
        "vllm_gpu_memory_utilization": 0.7,
        "use_vllm": False,
        "use_4bit": True
    }
}

for key in GRPO_CONFIG:
    GRPO_CONFIG[key]["label"] = key


def if_contain_slow_reward_function(dataset_type: dict) -> bool:
    reward_functions = dataset_type["reward_functions"]
    for reward_func in reward_functions:
        func_def = reward_func["reward_func"]
        keywords  = ["import langcheck", "from langcheck", "import detoxify", "from detoxify", "import textstat", "from textstat"]
        if any(keyword in func_def for keyword in keywords):
            return True
    return False
    

def get_grpo_config(param_nums: int) -> dict:
    if param_nums < 1_000_000_000:
        return GRPO_CONFIG["0_1_b"]
    elif param_nums < 2_000_000_000:
        return GRPO_CONFIG["1_2_b"]
    elif param_nums < 4_000_000_000:
        return GRPO_CONFIG["2_4_b"]
    elif param_nums < 5_000_000_000:
        return GRPO_CONFIG["4_5_b"]
    elif param_nums < 6_000_000_000:
        return GRPO_CONFIG["5_6_b"]
    elif param_nums < 9_000_000_000:
        return GRPO_CONFIG["6_9_b"]
    elif param_nums < 12_000_000_000:
        return GRPO_CONFIG["9_12_b"]
    elif param_nums < 15_000_000_000:  
        return GRPO_CONFIG["12_15_b"]
    elif param_nums < 20_000_000_000:
        return GRPO_CONFIG["15_20_b"]
    elif param_nums < 40_000_000_000:
        return GRPO_CONFIG["20_40_b"]
    elif param_nums < 80_000_000_000:
        return GRPO_CONFIG["40_80_b"]
    else:
        print(f"Model size {param_nums} is not supported")
        return {
            "lr": 4e-5,
            "distributed": "ds",
            "gpu_count": 8,
            "batch_size": 6,
            "use_lora": True
        }


def get_run_cmd(config: dict, gpu_nums: int):
    required_keys = [
        "epoch_num",
        "batch_size",
        "learning_rate",
        "min_lr_rate",
        "use_liger",
        "optimizer",
        "vllm_gpu_memory_utilization",
        "num_generations",
        "disable_fa",
    ]
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Required key {key} not found in config")

    start_cmd = "python"
    run_type = config["distributed"]
    # if gpu_nums > 1 and run_type == "ddp":
    gpu_nums = get_gpu_count()
    start_cmd = f"torchrun --nproc_per_node={gpu_nums}"
    if run_type == "ds":
        start_cmd = f"deepspeed"

    template = (
        start_cmd
        + """ train_grpo.py \
    --request_path {request_path} \
    --bf16 True \
    --report_to wandb \
    --output_dir /workspace/data/trained_model \
    --num_train_epochs {epoch_num} \
    --per_device_train_batch_size {batch_size} \
    --per_device_eval_batch_size {eval_batch_size} \
    --gradient_accumulation_steps {gradient_accumulation_steps} \
    --eval_accumulation_steps 1 \
    --eval_strategy no \
    --save_strategy no \
    --logging_steps 5 \
    --learning_rate {learning_rate} \
    --weight_decay 0. \
    --warmup_steps 35 \
    --lr_scheduler_type cosine_with_min_lr \
    --lr_scheduler_kwargs "{\\"min_lr_rate\\": {min_lr_rate}}" \
    --tf32 True \
    --gradient_checkpointing {gradient_checkpointing} \
    --optim {optimizer} \
    --use_liger {use_liger} --num_generations {num_generations} --vllm_mode colocate --vllm_gpu_memory_utilization {vllm_gpu_memory_utilization} \
    --disable_fa {disable_fa}"""
    )
    
    if config.get("use_lora", False):
        template += (
            " --use_peft --lora_r 128 --lora_alpha 256 --lora_target_modules all-linear"
        )
    
    if config.get("use_vllm", True):
        template += " --use_vllm True"
    else:
        template += " --use_vllm False"
    
    if run_type == "ds":
        template = template + """ --deepspeed ds_config/zero3.json"""

    for key, value in config.items():
        template = template.replace("{" + key + "}", str(value))
    
    if config.get("tensor_parallel", False):
        template = template + f" --vllm_tensor_parallel_size {gpu_nums}"
    
    if config.get("use_4bit", False):
        template = template + " --load_in_4bit True --use_bnb_nested_quant True --bnb_4bit_quant_type nf4"
    return template


def get_training_json(train_info: dict) -> dict:
    model_name = train_info["model_name"]
    model_path = train_info["model_path"]
    model_architecture = get_model_architecture(model_path)
    param_nums = get_model_num_params(model_name, model_path)
    config = get_grpo_config(param_nums)
    print(f"config: {config}")
    run_config = {
        "epoch_num": 2,
        "batch_size": config["batch_size"],
        "learning_rate": config["lr"],
        "min_lr_rate": 0.25,
        "use_liger": get_use_liger(model_architecture),
        "optimizer": "paged_adamw_8bit",
        "use_lora": config.get("use_lora", False),
        "disable_fa": disable_flash_attention(model_architecture, model_name),
        "gpu_nums": config["gpu_count"],
        "output_dir": train_info["output_dir"],
        "request_path": train_info["request_path"],
        "distributed": config.get("distributed", "ddp"),
        "gradient_checkpointing": get_gradient_checkpointing(model_name),
        "gradient_accumulation_steps": 4,
        "vllm_gpu_memory_utilization": config.get("vllm_gpu_memory_utilization", 0.4),
        "num_generations": 2,
        "use_vllm": get_use_vllm(model_architecture, model_name),
        "tensor_parallel": config.get("tensor_parallel", False),
        "use_4bit": config.get("use_4bit", False)
    }
    
    if model_name == "OpenAssistant/oasst-sft-4-pythia-12b-epoch-3.5":
        run_config["use_lora"] = True
    
    if "starcoder" in model_name.lower():
        run_config["batch_size"] = int(run_config["batch_size"] / 1.5)
    
    train_request = deepcopy(train_info)
    train_request["save_before_remaining_time"] = 3
    train_request["min_steps"] = 100
    train_request["adjust_batch_size"] = False
    train_request["periodic_save_steps"] = 500
    
    
    if if_contain_slow_reward_function(train_info["dataset_type"]):
        train_request["save_before_remaining_time"] = 12
        if config["label"] == "0_1_b":
            run_config["batch_size"] = 8
        elif config["label"] == "1_2_b":
            run_config["batch_size"] = 10
        elif config["label"] == "2_4_b":
            run_config["batch_size"] = 16
        elif config["label"] == "4_5_b":
            run_config["batch_size"] = 16
        elif config["label"] == "5_6_b":
            run_config["batch_size"] = 16
        elif config["label"] == "6_9_b":
            run_config["batch_size"] = 16
            if model_name == "unsloth/gemma-2-9b-it": # encounter OOM error with batch_size 12
                run_config["batch_size"] = 8
        elif config["label"] == "9_12_b":
            run_config["batch_size"] = 16
        elif config["label"] == "12_15_b":
            run_config["batch_size"] = 2
        elif config["label"] == "15_20_b":
            run_config["batch_size"] = 2
        elif config["label"] == "20_40_b":
            run_config["batch_size"] = 16 # this is high because we use 4bit
        elif config["label"] == "40_80_b":
            run_config["batch_size"] = 2
            
                
        elif config["label"] == "13_15_b":
            run_config["batch_size"] = 12
    
    total_batch_size = run_config["batch_size"] * run_config["gpu_nums"]
    if total_batch_size < 64:
        run_config["gradient_accumulation_steps"] = min(4, int(64 / total_batch_size))
    
    run_config["eval_batch_size"] = 4
    if run_config["batch_size"] <= 4:
        run_config["eval_batch_size"] = 2
    
    if not config.get("use_vllm", True):
        run_config["use_vllm"] = False

    if model_name in ["unsloth/gemma-2-9b-it"]:
        run_config["learning_rate"] = 1e-5
    if model_name in ["unsloth/codegemma-7b-it", "unsloth/gemma-1.1-2b-it"]:
        run_config["learning_rate"] = 2.5e-5
    if model_name in ["zake7749/gemma-2-2b-it-chinese-kyara-dpo", "unsloth/codegemma-2b"]:
        run_config["learning_rate"] = 1e-5
    if model_name in ["unsloth/codegemma-7b", "unsloth/gemma-7b-it"]:
        run_config["learning_rate"] = 8e-6
        
    run_cmd = get_run_cmd(run_config, run_config["gpu_nums"])
    
    return {
        "train_request": train_request,
        "run_cmd": run_cmd
    }