"""
Merge LoRA adapter into base model for a single, portable model.

Usage:
    python merge_adapter.py --adapter_path ./qwen-test --output_path ./qwen-epi-forecast-merged

    # Use different adapter (e.g. from full training)
    python merge_adapter.py --adapter_path ./qwen-epi-forecast --output_path ./qwen-epi-forecast-merged
"""

import argparse
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


def merge_and_save(adapter_path: str, output_path: str, base_model: str = "Qwen/Qwen2.5-3B-Instruct"):
    """Merge LoRA adapter into base model and save as single model."""
    adapter_path = Path(adapter_path)
    output_path = Path(output_path)

    if not adapter_path.exists():
        raise FileNotFoundError(f"Adapter path not found: {adapter_path}")

    print(f"Loading base model: {base_model}")
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.float16,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True,
    )

    print(f"Loading LoRA adapter from {adapter_path}")
    model = PeftModel.from_pretrained(model, adapter_path)

    print("Merging LoRA weights into base model...")
    model = model.merge_and_unload()

    output_path.mkdir(parents=True, exist_ok=True)
    print(f"Saving merged model to {output_path}")
    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)

    print(f"Done! Merged model saved to {output_path}")
    return str(output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--adapter_path",
        type=str,
        default="./qwen-test",
        help="Path to LoRA adapter (e.g. ./qwen-test or ./qwen-epi-forecast)",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="./qwen-epi-forecast-merged",
        help="Output path for merged model",
    )
    parser.add_argument(
        "--base_model",
        type=str,
        default="Qwen/Qwen2.5-3B-Instruct",
        help="Base model name",
    )
    args = parser.parse_args()

    merge_and_save(args.adapter_path, args.output_path, args.base_model)
