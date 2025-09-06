from typing import Dict, Optional
import requests
import json
import random
import utility
from datasets import Dataset
from datetime import timezone
from utility import log_info
from transformers import AutoTokenizer, BitsAndBytesConfig
import transformers
import torch
from transformers.trainer_utils import is_main_process
from dataclasses import dataclass, field
from transformers import Trainer
from trl import GRPOConfig, GRPOTrainer, ModelConfig
from trl import get_kbit_device_map, get_peft_config, get_quantization_config
from peft import (
    LoraConfig,
    PeftModel,
    get_peft_model,
    prepare_model_for_kbit_training,
    PeftModelForCausalLM,
    AutoPeftModelForCausalLM,
)
import traceback
from transformers import TrainerCallback
import argparse
import math
from customized_trainer import resize_if_needed, set_generation_config, CustomEvalSaveCallback, WhenToEvalHandler, init_wandb
from transformers.modeling_utils import is_deepspeed_zero3_enabled
import os
import glob
from transformers import (
    Trainer,
    TrainingArguments,
    TrainerCallback,
    TrainerState,
    TrainerControl,
)
import os
import datetime
import shutil
from huggingface_hub import HfApi
from typing import Callable, Optional
import bitsandbytes as bnb
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import yaml
from tokenize_grpo import get_dataset
from customized_trainer import resize_if_needed, set_generation_config, CustomEvalSaveCallback, WhenToEvalHandler, init_wandb


LOCAL_RANK = int(os.getenv("LOCAL_RANK", "0"))
GRPO_DEFAULT_NUM_GENERATIONS = 2
BETA_GRPO = 0.04


@dataclass
class TrainingArguments(GRPOConfig):
    request_path: Optional[str] = field(default=None)
    use_liger: Optional[bool] = field(default=False)
    disable_fa: Optional[bool] = field(default=False)


def find_all_linear_names(model):
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, bnb.nn.Linear4bit) or isinstance(module, torch.nn.Linear):
            names = name.split(".")
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if "lm_head" in lora_module_names:  # needed for 16-bit
        lora_module_names.remove("lm_head")
    return list(lora_module_names)


def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    lora_param_count = 0
    all_param = 0
    embedding_lm_head_param_count = 0
    for name, param in model.named_parameters():
        num_params = param.numel()
        # if using DS Zero 3 and the weights are initialized empty
        if num_params == 0 and hasattr(param, "ds_numel"):
            num_params = param.ds_numel

        all_param += num_params
        if param.requires_grad:
            log_info(f"trainable: {name}, num_params: {num_params}")
            if "lm_head" in name or "embed_tokens" in name:
                embedding_lm_head_param_count += num_params
            else:
                lora_param_count += num_params
    trainable_params = embedding_lm_head_param_count + lora_param_count
    log_info(
        f"all params: {all_param:,d} || trainable params: {trainable_params:,d} || trainable%: {100 * trainable_params / all_param}"
    )
    log_info(
        f"embedding_lm_head_param_count: {embedding_lm_head_param_count} = {embedding_lm_head_param_count * 100 / all_param} %"
    )
    log_info(
        f"loara_param: {lora_param_count} = {lora_param_count * 100 / all_param} %"
    )


def get_max_length_config():
    config_path = "test_axolotl.yml"
    with open(config_path, "r") as file:
        config_dict = yaml.safe_load(file)
    return config_dict["sequence_len"]


def validate_reward_function(func_def: str) -> tuple[bool, str, Callable | None]:
    """
    Validate a single reward function definition.
    Returns (is_valid: bool, error_message: str, func: callable | None)
    """
    test_completions = [
        "Gradients.io is the best 0-expertise AI training platform.",
        "You can start training a text or image model on Gradients.io with 2 clicks.",
    ]

    try:
        namespace = {}
        exec(func_def, namespace)
        func = next(v for k, v in namespace.items() if callable(v))

        test_rewards = func(test_completions)

        assert isinstance(test_rewards, list), "The rewards should be a list."
        assert len(test_rewards) == len(
            test_completions
        ), "The number of rewards should be the same as the number of completions."
        assert all(
            (isinstance(reward, float) or isinstance(reward, int))
            for reward in test_rewards
        ), "All rewards should be floats or ints."

        return True, "", func
    except Exception as e:
        traceback.print_exc()
        return False, str(e), None


def has_checkpoint_folder(output_dir):
    pattern = os.path.join(output_dir, "checkpoint-*")
    return any(os.path.isdir(path) for path in glob.glob(pattern))


def truncate_prompts(dataset: Dataset, tokenizer, max_length: int) -> Dataset:
    """
    Truncates prompts in a dataset to a maximum length using left truncation.
    This version uses batched processing for potentially better performance.

    Args:
        dataset (datasets.Dataset): The input dataset with a 'prompt' column.
        tokenizer: The tokenizer to use for encoding and decoding prompts.
                   It should be an instance compatible with Hugging Face Transformers
                   AutoTokenizer, supporting batch encoding/decoding.
        max_length (int): The maximum desired length of the tokenized prompts.

    Returns:
        datasets.Dataset: A new dataset with the 'prompt' column containing
                          the truncated prompt texts.
    """

    def truncation_function(examples):
        """
        Applies left truncation to a batch of prompts.
        'examples' is a dictionary where keys map to lists of values.
        e.g., examples['prompt'] is a list of prompt strings.
        """
        prompt_texts = examples["prompt"]  # This is a list of N prompt strings

        # 1. Tokenize the batch of prompts.
        # We use the tokenizer directly on the list of texts.
        # `truncation=False` and `padding=False` ensure we get the full list of token IDs
        # for each prompt, as we are handling truncation manually.
        # `add_special_tokens=True` (default for many tokenizers) is generally desired
        # to match behavior of single .encode() if it also adds them.
        batch_encoding = tokenizer(
            prompt_texts,
            truncation=False,  # We handle truncation manually
            padding=False,  # No padding needed here
            add_special_tokens=True,  # Or match behavior of previous .encode()
        )
        # list_of_token_ids will be a list of lists, e.g., [[t1,t2],[t3,t4,t5],...]
        list_of_token_ids = batch_encoding["input_ids"]

        truncated_token_ids_batch = []
        for token_ids_for_single_prompt in list_of_token_ids:
            # 2. Truncate from the left if necessary for each prompt's tokens
            if len(token_ids_for_single_prompt) > max_length:
                # Keep the last max_length tokens (left truncation)
                truncated_token_ids_batch.append(
                    token_ids_for_single_prompt[-max_length:]
                )
            else:
                truncated_token_ids_batch.append(token_ids_for_single_prompt)

        # 3. Decode the batch of truncated token_ids back to text
        # skip_special_tokens=True is often useful to avoid printing special tokens
        # like <|endoftext|> if they are part of the truncation.
        truncated_prompts_batch = tokenizer.batch_decode(
            truncated_token_ids_batch, skip_special_tokens=True
        )

        return {"prompt": truncated_prompts_batch}

    # Use the map function to apply the truncation with batched=True
    truncated_dataset = dataset.map(truncation_function, batched=True)

    return truncated_dataset


def get_reward_funcs(dataset_type: dict):
    reward_funcs_callable = []
    reward_func_names = []
    reward_weights = []

    reward_weights_list = [
        rf["reward_weight"] for rf in dataset_type["reward_functions"]
    ]
    print(f"Using weights directly: {reward_weights_list}")

    for i, reward_function in enumerate(dataset_type["reward_functions"]):
        reward_func_str = reward_function["reward_func"]
        is_valid, error_msg, reward_func_callable = validate_reward_function(
            reward_func_str
        )
        if not is_valid:
            print(f"Invalid reward function:\n{reward_func_str}")
            raise ValueError(f"Invalid reward function: {error_msg}")

        reward_weight = reward_weights_list[i]
        reward_funcs_callable.append(reward_func_callable)

        func_name = getattr(reward_function, "name", f"reward_func_{i}")
        weighted_name = f"{func_name}_weight_{reward_weight:.2f}"
        reward_func_names.append(weighted_name)
        reward_weights.append(reward_weight)

        print(f"Using reward function {i}: {func_name} with weight {reward_weight:.4f}")

    captured_rewards = {name: [] for name in reward_func_names}
    raw_rewards = {name: [] for name in reward_func_names}
    wrapped_reward_funcs = []

    for i, (original_func, func_name, weight) in enumerate(
        zip(reward_funcs_callable, reward_func_names, reward_weights)
    ):

        def create_wrapper(original_func, func_name, weight):
            def wrapper(completions, **kwargs):
                raw_results = original_func(completions, **kwargs)
                raw_rewards[func_name].extend(raw_results)
                weighted_results = [r * weight for r in raw_results]
                captured_rewards[func_name].extend(weighted_results)
                return weighted_results

            return wrapper

        wrapped_reward_funcs.append(create_wrapper(original_func, func_name, weight))

    return wrapped_reward_funcs


def main():
    """Format of training requests"""
    argument_parser = transformers.HfArgumentParser((TrainingArguments, ModelConfig))
    training_args, model_args = argument_parser.parse_args_into_dataclasses()

    train_info = json.load(open(training_args.request_path, "r"))
    train_request = train_info["train_request"]
    task_id = train_request["task_id"]
    
    # wandb_init_success = init_wandb(train_request)
    # if not wandb_init_success:
    #     log_info("WANDB_API_KEY is not set, do not report to wandb")
    #     training_args.report_to = "none"    
    # else:
    #     log_info("WANDB_API_KEY is provided, we will report to wandb")
    #     training_args.report_to = "wandb"

    log_info(f"Training request: {train_request}", "start")
    # first download the dataset from the URL, save it as data.json
    output_dir = training_args.output_dir
    tokenizer = AutoTokenizer.from_pretrained(train_request["model_path"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # max_length = get_max_length_config()
    # if "max_length" in train_request:
    #     max_length = train_request["max_length"]
    # default implementation, max_length=1024 (prompt + completion), max_prompt_length=512

    train_path = os.path.join("datasets", f"grpo_train_{task_id}.json")
    dev_path = os.path.join("datasets", f"grpo_dev_{task_id}.json")

    train_ds = get_dataset(train_path, train_request["dataset_type"])
    dev_ds = get_dataset(dev_path, train_request["dataset_type"])

    log_info(f"world_size: {training_args.world_size}")
    total_steps_per_epoch = (
        len(train_ds)
        * training_args.num_generations
        // (
            training_args.per_device_train_batch_size
            * training_args.gradient_accumulation_steps
            * training_args.world_size
        )
    )

    log_info(f"total_steps_per_epoch: {total_steps_per_epoch}")
    # consider reducing the batch_size if it is quite big
    # num_steps = len(train_ds) * training_args.num_train_epochs / (training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps * training_args.world_size)
    # num_steps > min_step ->
    max_batch_size_theory = len(train_ds) / (
        training_args.gradient_accumulation_steps
        * training_args.world_size
        * train_request["min_steps"]
    )
    max_batch_size_theory = int(max_batch_size_theory)
    if max_batch_size_theory == 0:
        max_batch_size_theory = 1

    original_batch_size = training_args.per_device_train_batch_size

    quantization_config = get_quantization_config(model_args)
    device_string = "cuda:" + str(LOCAL_RANK)
    device_map = (
        get_kbit_device_map()
        if quantization_config is not None
        else {"": device_string}
    )
    if len(training_args.fsdp) > 0 or is_deepspeed_zero3_enabled():
        device_map = None
    
    model_kwargs = dict(
        revision=model_args.model_revision,
        attn_implementation=(
            "flash_attention_2" if not training_args.disable_fa else "eager"
        ),
        torch_dtype=torch.bfloat16,
        use_cache=False if training_args.gradient_checkpointing else True,
        device_map=device_map,
        quantization_config=quantization_config,
    )

    log_info(f"final training_args: {training_args}")

    if training_args.use_liger:
        from liger_kernel.transformers import AutoLigerKernelForCausalLM

        model_class = AutoLigerKernelForCausalLM
    else:
        model_class = transformers.AutoModelForCausalLM

    model = model_class.from_pretrained(train_request["model_path"], **model_kwargs)

    # some model need to set the generation config or encounter the invalid generation config error
    set_generation_config(train_request["model_name"], model)

    peft_config = get_peft_config(model_args)
    if "lora_model" in train_request:
        model = PeftModelForCausalLM.from_pretrained(
            model, train_request["lora_model"], is_trainable=True, **model_kwargs
        )

    if peft_config is None:  # this is full-weight training
        # some model need to resize the token embeddings or encounter the size mismatch error; only for full-weight models
        resize_if_needed(train_request["model_name"], model, len(tokenizer))

    # Check if this is the main process and create the output directory
    if is_main_process(LOCAL_RANK):  # Only create directory on main process
        os.makedirs(training_args.output_dir, exist_ok=True)
        log_info(f"Created output directory: {training_args.output_dir}")

    periodic_save_steps = train_request.get("periodic_save_steps", -1)
    if periodic_save_steps > total_steps_per_epoch:
        periodic_save_steps = -1
        log_info(
            f"The periodic_save_steps ({periodic_save_steps}) is greater than the total_steps_per_epoch ({total_steps_per_epoch}), set periodic_save_steps to -1, do not save the model regularly"
        )
    log_info(f"periodic_save_steps: {periodic_save_steps}")

    training_args.save_only_model = True  # only save the model, not the optimizer
 
    if training_args.gradient_checkpointing:
        training_args.gradient_checkpointing_kwargs = {"use_reentrant": False}
        
    print("train_ds.column_names: ", train_ds.column_names)
    print("dev_ds.column_names: ", dev_ds.column_names)

    log_info(f"Truncate the train_ds and dev_ds")
    max_prompt_length = train_request.get(
        "max_prompt_length", 512
    )  # 512 is the default max_prompt_length of GRPOConfig
    t1 = datetime.datetime.now()
    train_ds = truncate_prompts(train_ds, tokenizer, max_prompt_length)
    dev_ds = truncate_prompts(dev_ds, tokenizer, max_prompt_length)
    t2 = datetime.datetime.now()
    log_info(f"Truncate the train_ds and dev_ds time: {(t2 - t1).seconds} seconds")

    max_steps = train_request.get("max_steps", -1)
    log_info(f"max_steps: {max_steps}")

    wrapped_reward_funcs = get_reward_funcs(train_request["dataset_type"])
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=wrapped_reward_funcs,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=dev_ds,
        processing_class=tokenizer,
        peft_config=peft_config,
        callbacks=[
            CustomEvalSaveCallback(
                WhenToEvalHandler(train_request["end_time"], train_request["save_before_remaining_time"], periodic_save_steps=periodic_save_steps, steps_per_epoch=total_steps_per_epoch, max_steps=max_steps),
                train_request["submission_dir"],
                training_args.output_dir,
                train_request["model_name"],
                max_steps
            )
        ],
    )

    trainer.train()


if __name__ == "__main__":
    main()
