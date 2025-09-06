from pydantic import BaseModel, Field
from enum import Enum


class InstructTextDatasetType(BaseModel):
    system_prompt: str | None = ""
    system_format: str | None = "{system}"
    field_system: str | None = None
    field_instruction: str | None = None
    field_input: str | None = None
    field_output: str | None = None
    format: str | None = None
    no_input_format: str | None = None
    field: str | None = None


class DpoDatasetType(BaseModel):
    field_prompt: str | None = None
    field_system: str | None = None
    field_chosen: str | None = None
    field_rejected: str | None = None
    prompt_format: str | None = "{prompt}"
    chosen_format: str | None = "{chosen}"
    rejected_format: str | None = "{rejected}"


class RewardFunction(BaseModel):
    """Model representing a reward function with its metadata"""

    reward_func: str = Field(
        ...,
        description="String with the python code of the reward function to use",
        examples=[
            "def reward_func_conciseness(completions, **kwargs):",
            '"""Reward function that favors shorter, more concise answers."""',
            "    return [100.0/(len(completion.split()) + 10) for completion in completions]",
        ],
    )
    reward_weight: float = Field(..., ge=0)
    func_hash: str | None = None
    is_generic: bool | None = None


class GrpoDatasetType(BaseModel):
    field_prompt: str | None = None
    reward_functions: list[RewardFunction] | None = []


class ChatTemplateDatasetType(BaseModel):
    chat_template: str | None = "chatml"
    chat_column: str | None = "conversations"
    chat_role_field: str | None = "from"
    chat_content_field: str | None = "value"
    chat_user_reference: str | None = "user"
    chat_assistant_reference: str | None = "assistant"



TextDatasetType = InstructTextDatasetType | DpoDatasetType | GrpoDatasetType | ChatTemplateDatasetType


class FileFormat(str, Enum):
    CSV = "csv"  # needs to be local file
    JSON = "json"  # needs to be local file
    HF = "hf"  # Hugging Face dataset
    S3 = "s3"

class TrainRequest(BaseModel):
    model: str = Field(..., description="Name or path of the model to be trained", min_length=1)
    task_id: str
    hours_to_complete: float
    expected_repo_name: str | None = None


class TrainRequestText(TrainRequest):
    dataset: str = Field(
        ...,
        description="Path to the dataset file or Hugging Face dataset name",
        min_length=1,
    )
    dataset_type: TextDatasetType
    file_format: FileFormat


class TrainerProxyRequest(BaseModel):
    training_data: TrainRequestText
    github_repo: str
    gpu_ids: list[int]
    hotkey: str
    github_branch: str | None = None
    github_commit_hash: str | None = None


class TaskType(str, Enum):
    INSTRUCTTEXTTASK = "InstructTextTask"
    IMAGETASK = "ImageTask"
    DPOTASK = "DpoTask"
    GRPOTASK = "GrpoTask"
    CHATTASK = "ChatTask"

    def __hash__(self):
        return hash(str(self))
    

def get_task_type(request: TrainerProxyRequest) -> TaskType:
    training_data = request.training_data

    if isinstance(training_data, TrainRequestText):
        if isinstance(training_data.dataset_type, DpoDatasetType):
            return TaskType.DPOTASK
        elif isinstance(training_data.dataset_type, InstructTextDatasetType):
            return TaskType.INSTRUCTTEXTTASK
        elif isinstance(training_data.dataset_type, GrpoDatasetType):
            return TaskType.GRPOTASK
        else:
            raise ValueError(f"Unsupported dataset_type for text task: {type(training_data.dataset_type)}")

    raise ValueError(f"Unsupported training_data type: {type(training_data)}")


class LogContext:
    def __init__(self, **tags: str | dict):
        self.tags = tags
        self.token = None

    def __enter__(self):
        try:
            current = current_context.get()
            new_context = {**current, **self.tags}
        except LookupError:
            new_context = self.tags
        self.token = current_context.set(new_context)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.token:
            current_context.reset(self.token)
