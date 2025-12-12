"""
Merge LoRA Adapter with Base Model - Google Colab Version

Run this in Google Colab to merge your LoRA adapter with the base model
and upload to HuggingFace Hub.

Usage in Colab:
    1. Upload your adapter folder to Colab (or clone from HF Hub)
    2. Run this script with the appropriate parameters
"""

# CONFIGURATION - EDIT THESE VALUES

# Your adapter path (local folder or HuggingFace Hub ID)
ADAPTER_PATH = "dtp-fine-tuning/multi-turn_chatbot_diploy"  
BASE_MODEL = "aitfindonesia/Bakti-8B-Base"

# Output settings
OUTPUT_PATH = "./merged_model"
HUB_MODEL_ID = "dtp-fine-tuning/multi-turn_chatbot_diploy"  # Same repo to overwrite

# Model settings
DTYPE = "bfloat16"  # or "float16"
MAX_SEQ_LENGTH = 2048

# HuggingFace Token (or set via huggingface-cli login)
HF_TOKEN = None  # or "hf_xxxxx"

# INSTALLATION (run this cell first in Colab)
"""
!pip install -q torch transformers accelerate peft huggingface_hub
"""

# MERGE SCRIPT

import os
import json
import torch
from pathlib import Path


def get_dtype(dtype_str: str) -> torch.dtype:
    """Convert string dtype to torch.dtype"""
    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    return dtype_map[dtype_str]


def get_base_model_from_adapter(adapter_path: str) -> str:
    """Read base model name from adapter_config.json"""
    # Check if it's a HF Hub path
    if "/" in adapter_path and not os.path.exists(adapter_path):
        from huggingface_hub import hf_hub_download
        config_path = hf_hub_download(
            repo_id=adapter_path,
            filename="adapter_config.json"
        )
    else:
        config_path = os.path.join(adapter_path, "adapter_config.json")
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    return config.get("base_model_name_or_path")


def merge_and_upload(
    adapter_path: str,
    output_path: str,
    hub_model_id: str,
    base_model: str = None,
    dtype: str = "bfloat16",
    hf_token: str = None,
):
    """Merge LoRA adapter with base model and upload to HuggingFace Hub"""
    
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    from huggingface_hub import HfApi, login
    
    # Login to HuggingFace
    if hf_token:
        login(token=hf_token)
    
    # Get base model name
    if base_model is None:
        print("Reading base model from adapter_config.json...")
        base_model = get_base_model_from_adapter(adapter_path)
    
    print("=" * 60)
    print("LoRA Merge Configuration")
    print("=" * 60)
    print(f"Base model:    {base_model}")
    print(f"Adapter path:  {adapter_path}")
    print(f"Output path:   {output_path}")
    print(f"Hub model ID:  {hub_model_id}")
    print(f"Data type:     {dtype}")
    print("=" * 60)
    
    # Get torch dtype
    torch_dtype = get_dtype(dtype)
    
    # Load base model
    print(f"\n[1/5] Loading base model: {base_model}")
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch_dtype,
        device_map="auto",
        trust_remote_code=True,
    )
    
    # Load tokenizer
    print(f"\n[2/5] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        base_model,
        trust_remote_code=True,
    )
    
    # Load LoRA adapter
    print(f"\n[3/5] Loading LoRA adapter: {adapter_path}")
    model = PeftModel.from_pretrained(model, adapter_path)
    
    # Merge adapter with base model
    print(f"\n[4/5] Merging adapter with base model...")
    model = model.merge_and_unload()
    
    # Save merged model locally
    print(f"\n[5/5] Saving merged model to: {output_path}")
    os.makedirs(output_path, exist_ok=True)
    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)
    
    # Create training info
    training_info = {
        "merged_from": {
            "base_model": base_model,
            "adapter_path": adapter_path,
        },
        "merge_type": "LoRA merge_and_unload",
        "dtype": dtype,
    }
    
    with open(os.path.join(output_path, "training_info.json"), 'w') as f:
        json.dump(training_info, f, indent=2)
    
    # Upload to HuggingFace Hub
    print(f"\nUploading to HuggingFace Hub: {hub_model_id}")
    api = HfApi()
    api.upload_folder(
        folder_path=output_path,
        repo_id=hub_model_id,
        repo_type="model",
        commit_message="Upload merged model (LoRA merged with base)"
    )
    
    print("\n" + "=" * 60)
    print("✓ Merge and upload complete!")
    print("=" * 60)
    print(f"Model available at: https://huggingface.co/{hub_model_id}")
    
    return model, tokenizer


# MAIN EXECUTION
if __name__ == "__main__":
    merge_and_upload(
        adapter_path=ADAPTER_PATH,
        output_path=OUTPUT_PATH,
        hub_model_id=HUB_MODEL_ID,
        base_model=BASE_MODEL,
        dtype=DTYPE,
        hf_token=HF_TOKEN,
    )
