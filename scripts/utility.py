from typing import Dict, Any
import yaml
from torch.utils.data import Dataset
from pathlib import Path
from transformers import AutoTokenizer
from typing import Callable
import torch
import logging
from datetime import datetime
import sys
import wandb
import random
import json
import requests
import os
import shutil
from transformers.trainer_utils import is_main_process

logger = logging.getLogger()
logger.setLevel(logging.INFO)
 # Create console handler
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)
 # Create formatter
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)
LOCAL_RANK = int(os.getenv("LOCAL_RANK", "0"))



def log_info(message: str, event_name: str = "print"):
    if is_main_process(LOCAL_RANK):
        logger.info(f"{event_name}: {message}")
    # wandb.log({"event": event_name, "message": message})


def pad_sequence(sequence: list[int], pad_value: int, max_length: int, padding_side: str) -> list[int]:
    if padding_side == "left":
        return [pad_value] * (max_length - len(sequence)) + sequence
    else:
        return sequence + [pad_value] * (max_length - len(sequence))


def pad_inputs(tokenizer: AutoTokenizer, input_dict: dict, max_length: int, padding_side: str) -> dict:
    assert padding_side in ["left", "right"]
    result = {
        "input_ids": pad_sequence(input_dict["input_ids"], tokenizer.pad_token_id, max_length, padding_side),
        "attention_mask": pad_sequence(input_dict["attention_mask"], 0, max_length, padding_side),
        "labels": pad_sequence(input_dict["labels"], -100, max_length, padding_side),
    }
    return result


class MyDataset(Dataset):
    def __init__(self, tokenizer: AutoTokenizer, data_path: str, max_length: int) -> None:
        super().__init__()
        with open(data_path, 'r') as file:
            self.eval_dataset = json.load(file)
            
        self.tokenizer = tokenizer
        self.max_length = max_length
        print("padding_side: ", self.tokenizer.padding_side)
        
    def __len__(self):
        return len(self.eval_dataset)
    
    def __getitem__(self, idx):
        dp = self.eval_dataset[idx]
        input_dict = pad_inputs(self.tokenizer, dp, self.max_length, self.tokenizer.padding_side)
        for key in input_dict:
            input_dict[key] = torch.tensor(input_dict[key])
        return input_dict