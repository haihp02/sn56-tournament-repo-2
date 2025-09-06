import typer 
import json 
from datasets import Dataset, load_dataset
import random 
import os
TRL_DPO_FIELD_PROMPT = "prompt"
TRL_DPO_FIELD_CHOSEN = "chosen"
TRL_DPO_FIELD_REJECTED = "rejected"
BETA_DPO = 0.1
from datetime import datetime 


REMOVE_ADD_TOKEN = {
    "berkeley-nest/Starling-LM-7B-alpha": "<sep>",
    "NousResearch/Nous-Capybara-7B-V1": "<pad>",
    "NousResearch/Hermes-2-Theta-Llama-3-8B": "<tool_response>",
    "MNC-Jihun/Mistral-7B-AO-u0.5-b2-ver0.4": "[PAD]"
}


def stringify_wrong_item(items):
    for item in items:
        for k, v in item.items():
            if type(v) is not str:
                item[k] = str(v)
    return items


def remove_sep_token(items, sep_token: str): # model berkeley-nest/Starling-LM-7B-alpha don't accept <sep> token
    for item in items:
        for k in item:
            item[k] = item[k].replace(sep_token, "")
    return items


def is_poor_item(item):
    for key, value in item.items():
        if value is None or (type(value) is str and len(value.strip()) == 0):
            return True
    return False


def remove_empty_items(items: list):
    result = []
    count = 0
    for item in items:
        if not is_poor_item(item):
            result.append(item)
        else:
            count += 1
    print(f"Removed {count} empty items")
    return result


def split_dataset(total_data_path: str, train_data_path: str, dev_data_path: str, seed: int = 42, dev_size: int = 200, max_data_size: int = -1, model: str = ""):
    """Split the dataset into train and dev"""
    # Load the dataset
    with open(total_data_path, 'r') as file:
        data = json.load(file)
    
    random.seed(seed)
    random.shuffle(data)
    stringify_wrong_item(data)
    data = remove_empty_items(data)
    if model in REMOVE_ADD_TOKEN:
        print(f"Removing {REMOVE_ADD_TOKEN[model]} token from {model}")
        data = remove_sep_token(data, REMOVE_ADD_TOKEN[model])
    
    if max_data_size > 0:
        data = data[:max_data_size]
    
    # Split the dataset into train and dev
    dev_items = data[:dev_size]
    train_items = data[dev_size:]
    # Save the train and dev datasets
    with open(train_data_path, 'w') as file:
        json.dump(train_items, file, ensure_ascii=False)
    
    with open(dev_data_path, 'w') as file:
        json.dump(dev_items, file, ensure_ascii=False)
        
    print(f"split {total_data_path} ({len(data)} items) into {train_data_path} ({len(train_items)} items) and {dev_data_path} ({len(dev_items)} items)")


def _adapt_dpo_columns_to_trl(dataset: Dataset, dataset_type: dict) -> Dataset:
    """
    Transform a DPO dataset to match trl's expected column names.

    Args:
        dataset: Hugging Face dataset object
        dataset_type: DpoDatasetType with field mappings
    """
    print("Adapting DPO columns to standard format")

    chosen_field = dataset_type["field_chosen"]
    rejected_field = dataset_type["field_rejected"]
    
    if chosen_field in dataset.column_names and rejected_field in dataset.column_names:
        identical_count = 0
        sample_size = min(10, len(dataset))
        sample_indices = list(range(sample_size))
        
        for idx in sample_indices:
            example = dataset[idx]
            chosen = example[chosen_field]
            rejected = example[rejected_field]
            
            if chosen == rejected:
                identical_count += 1
        
        if identical_count > 0:
            print(f"CRITICAL: Found {identical_count}/{sample_size} samples with identical chosen/rejected, causing random predictions")

            if identical_count > 0:
                example = dataset[sample_indices[0]]
                chosen = example[chosen_field]
                rejected = example[rejected_field]
                print(f"Example: Chosen/Rejected: '{chosen[:100]}...'")

    column_mapping = {
        dataset_type["field_prompt"]: TRL_DPO_FIELD_PROMPT,
        dataset_type["field_chosen"]: TRL_DPO_FIELD_CHOSEN,
        dataset_type["field_rejected"]: TRL_DPO_FIELD_REJECTED
    }
    for src_col, dst_col in column_mapping.items():
        if src_col in dataset.column_names and src_col != dst_col:
            dataset = dataset.rename_column(src_col, dst_col)

    columns_to_keep = [TRL_DPO_FIELD_PROMPT, TRL_DPO_FIELD_CHOSEN, TRL_DPO_FIELD_REJECTED]
    columns_to_remove = [col for col in dataset.column_names if col not in columns_to_keep]
    for col in columns_to_remove:
        dataset = dataset.remove_columns(col)

    return dataset


def get_dataset(path: str, dataset_type:dict):
    eval_dataset = load_dataset("json", data_files=path, split="train")
    eval_dataset = _adapt_dpo_columns_to_trl(eval_dataset, dataset_type)
    return eval_dataset


def main(training_request_path: str):
    with open(training_request_path, 'r') as file:
        training_request = json.load(file)
    
    t1 = datetime.now()
    total_path = training_request["train_request"]["dataset"]
    task_id = training_request["train_request"]["task_id"]
    
    train_path = os.path.join("datasets", f"dpo_train_{task_id}.json")
    dev_path = os.path.join("datasets", f"dpo_dev_{task_id}.json")
    
    max_data_size = training_request["train_request"].get("max_data_size", -1)
    if max_data_size > 0:
        print(f"Max data size is {max_data_size}, so we will only extract {max_data_size} samples randomly")
    
    model_name = training_request["train_request"]["model_name"]
    
    split_dataset(total_path, train_path, dev_path, max_data_size=max_data_size, model=model_name)
    t2 = datetime.now()
    print(f"Tokenization completed in {(t2 - t1).seconds} seconds")


if __name__ == "__main__":
    typer.run(main)