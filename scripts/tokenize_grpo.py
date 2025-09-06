import typer 
import json 
from datasets import Dataset, load_dataset
import requests
import random 
import os
TRL_DPO_FIELD_PROMPT = "prompt"
TRL_DPO_FIELD_CHOSEN = "chosen"
TRL_DPO_FIELD_REJECTED = "rejected"
BETA_DPO = 0.1
from datetime import datetime 
import shutil

    

def stringify_wrong_item(items):
    for item in items:
        for k, v in item.items():
            if type(v) is not str:
                item[k] = str(v)
    return items


def split_dataset(total_data_path: str, train_data_path: str, dev_data_path: str, seed: int = 42, dev_size: int = 200):
    """Split the dataset into train and dev"""
    # Load the dataset
    with open(total_data_path, 'r') as file:
        data = json.load(file)
    
    random.seed(seed)
    random.shuffle(data)
    
    # Split the dataset into train and dev
    dev_items = data[:dev_size]
    train_items = data[dev_size:]
    
    # Save the train and dev datasets
    with open(train_data_path, 'w') as file:
        stringify_wrong_item(train_items)
        json.dump(train_items, file, ensure_ascii=False)
    
    with open(dev_data_path, 'w') as file:
        stringify_wrong_item(dev_items)
        json.dump(dev_items, file, ensure_ascii=False)
        
    print(f"split {total_data_path} ({len(data)} items) into {train_data_path} ({len(train_items)} items) and {dev_data_path} ({len(dev_items)} items)")


def _adapt_grpo_columns_to_trl(dataset: Dataset, dataset_type: dict) -> Dataset:
    """
    Transform a GRPO dataset to match trl's expected column names.

    Args:
        dataset: Hugging Face dataset object
        dataset_type: GrpoDatasetType with field mappings
    """
    print("Adapting GRPO columns to standard format")

    column_mapping = {
        dataset_type["field_prompt"]: "prompt",
    }
    for src_col, dst_col in column_mapping.items():
        if src_col in dataset.column_names and src_col != dst_col:
            dataset = dataset.rename_column(src_col, dst_col)

    return dataset

def get_dataset(path: str, dataset_type:dict):
    eval_dataset = load_dataset("json", data_files=path, split="train")
    eval_dataset = _adapt_grpo_columns_to_trl(eval_dataset, dataset_type)
    return eval_dataset


def main(training_request_path: str):
    with open(training_request_path, 'r') as file:
        training_request = json.load(file)
    
    t1 = datetime.now()
    total_path = training_request["train_request"]["dataset"]
    task_id = training_request["train_request"]["task_id"]
    train_path = os.path.join("datasets", f"grpo_train_{task_id}.json")
    dev_path = os.path.join("datasets", f"grpo_dev_{task_id}.json")
    split_dataset(total_path, train_path, dev_path)
    t2 = datetime.now()
    print(f"Tokenization completed in {(t2 - t1).seconds} seconds")


if __name__ == "__main__":
    typer.run(main)