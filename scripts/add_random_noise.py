import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import typer 
from transformers import GenerationConfig
from huggingface_hub import HfApi
from transformers import AutoConfig
import json 
import datetime
# Configuration

def main(model_path: str, save_folder: str):
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, device_map="auto")
    model.generation_config = GenerationConfig(temperature=None, top_p=None)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    noise_std = 0.01

    # Step 2: Add random noise to the input embeddings
    print("Modifying input embeddings...")
    with torch.no_grad():
        embeddings = model.get_input_embeddings()
        noise = torch.randn_like(embeddings.weight) * noise_std
        embeddings.weight.add_(noise)

    # Step 3: Save the modified model and tokenizer
    print(f"Saving modified model to {save_folder}...")
    os.makedirs(save_folder, exist_ok=True)
    model.save_pretrained(save_folder)
    tokenizer.save_pretrained(save_folder)

if __name__ == "__main__":
    typer.run(main)
