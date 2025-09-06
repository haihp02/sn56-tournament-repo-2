from typing import Dict, Optional
import requests
import json
import random
import utility
from datasets import Dataset
from utility import log_info
from transformers import AutoTokenizer, BitsAndBytesConfig
import transformers
import torch
from transformers.trainer_utils import is_main_process
from dataclasses import dataclass, field
from transformers import Trainer
from trl import DPOTrainer, DPOConfig, ModelConfig, ScriptArguments, TrlParser
from trl import get_kbit_device_map, get_peft_config, get_quantization_config
from peft import (
    LoraConfig,
    PeftModel,
    get_peft_model,
    prepare_model_for_kbit_training,
    PeftModelForCausalLM,
    AutoPeftModelForCausalLM,
)
from transformers import TrainerCallback
import argparse
import os
from customized_trainer import resize_if_needed, set_generation_config, CustomEvalSaveCallback, WhenToEvalHandler, init_wandb


# from packing.packed_dataset import PackedDataset
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
import yaml
from tokenize_dpo import get_dataset
from transformers.modeling_utils import is_deepspeed_zero3_enabled



LOCAL_RANK = int(os.getenv("LOCAL_RANK", "0"))


@dataclass
class TrainingArguments(DPOConfig):
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


def make_parser(subparsers: argparse._SubParsersAction = None):
    dataclass_types = (TrainingArguments, ModelConfig)
    if subparsers is not None:
        parser = subparsers.add_parser(
            "dpo", help="Run the DPO training script", dataclass_types=dataclass_types
        )
    else:
        parser = TrlParser(dataclass_types)
    return parser


def main():
    """Format of training requests"""
    parser = make_parser()
    training_args, model_args = parser.parse_args_and_config()
    train_info = json.load(open(training_args.request_path, "r"))
    train_request = train_info["train_request"]

    # check if need to run early stop or not
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

    train_path = os.path.join("datasets", f"dpo_train_{task_id}.json")
    dev_path = os.path.join("datasets", f"dpo_dev_{task_id}.json")

    train_ds = get_dataset(train_path, train_request["dataset_type"])
    dev_ds = get_dataset(dev_path, train_request["dataset_type"])

    log_info(f"world_size: {training_args.world_size}")
    total_steps_per_epoch = len(train_ds) // (
        training_args.per_device_train_batch_size
        * training_args.gradient_accumulation_steps
        * training_args.world_size
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
    
    device_map=(
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
    if len(training_args.fsdp) > 0 or is_deepspeed_zero3_enabled():
        # set gradient checkpointing to True with use_reentrant=True for deepspeed
        log_info("Setting gradient checkpointing to True with use_reentrant=True for deepspeed")
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': True})


    # some model need to set the generation config or encounter the invalid generation config error
    set_generation_config(train_request["model_name"], model)

    ref_model = None
    if "ref_model" in train_request:
        ref_model = model_class.from_pretrained(
            train_request["ref_model"], **model_kwargs
        )
        # print("load ref_model: ", train_request["ref_model"])

    peft_config = get_peft_config(model_args)
    if "lora_model" in train_request:
        model = PeftModelForCausalLM.from_pretrained(
            model, train_request["lora_model"], is_trainable=True, **model_kwargs
        )

    if peft_config is None:  # this is full-weight training
        # some model need to resize the token embeddings or encounter the size mismatch error; only for full-weight models
        resize_if_needed(train_request["model_name"], model, len(tokenizer))

    # Only resize token embeddings if not using LoRA
    # if peft_config is None:  # full-weights training
    #    model.resize_token_embeddings(len(tokenizer))

    # Check if this is the main process and create the output directory
    if is_main_process(LOCAL_RANK):  # Only create directory on main process
        os.makedirs(training_args.output_dir, exist_ok=True)
        log_info(f"Created output directory: {training_args.output_dir}")

    periodic_save_steps = train_request.get("periodic_save_steps", -1)
    log_info(f"periodic_save_steps: {periodic_save_steps}")

    training_args.save_only_model = True  # only save the model, not the optimizer

    if training_args.gradient_checkpointing:
        training_args.gradient_checkpointing_kwargs = {"use_reentrant": False}
    print("train_ds.column_names: ", train_ds.column_names)

    max_steps = train_request.get("max_steps", -1)
    log_info(f"max_steps: {max_steps}")

    trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
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
    
    print("Start training ...")       
    trainer.train()


if __name__ == "__main__":
    main()