"""
Minimal training script that uses fewer packages.
Avoids heavy dependencies where possible.

Install only:
    pip install --no-cache-dir torch transformers peft datasets

Usage:
    python train_minimal_packages.py \
        --data_dir training_data \
        --output_dir ./qwen-epi-forecast
"""

import argparse
import json
import os
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model, TaskType


def load_and_format_dataset(data_dir, tokenizer):
    """Load and format dataset."""
    print("[1/4] Loading dataset...")
    data_files = {
        "train": str(Path(data_dir) / "train.jsonl"),
        "validation": str(Path(data_dir) / "val.jsonl"),
    }
    dataset = load_dataset("json", data_files=data_files)

    print(f"  Train: {len(dataset['train'])} examples")
    print(f"  Val:   {len(dataset['validation'])} examples")

    print("[2/4] Formatting with chat template...")

    def format_example(example):
        formatted = tokenizer.apply_chat_template(
            example["messages"],
            tokenize=False,
            add_generation_prompt=False
        )
        return {"text": formatted}

    dataset = dataset.map(format_example, remove_columns=dataset["train"].column_names)

    # Tokenize
    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, max_length=512, padding="max_length")

    tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])

    return tokenized_dataset


def main(args):
    print("=" * 80)
    print("Minimal Package Training (4 packages only)")
    print("=" * 80)
    print("\nPackages used: torch, transformers, peft, datasets")
    print("NOT using: trl, accelerate, bitsandbytes")
    print()

    # Load tokenizer
    print("[1/4] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        trust_remote_code=True,
        use_fast=True,
    )
    tokenizer.pad_token = tokenizer.eos_token

    # Load and format dataset
    tokenized_dataset = load_and_format_dataset(args.data_dir, tokenizer)

    # Load model
    print("[3/4] Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        trust_remote_code=True,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
        low_cpu_mem_usage=True,
    )

    # Add LoRA
    print("[4/4] Configuring LoRA...")
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_steps=100,
        logging_steps=10,
        save_strategy="epoch",
        eval_strategy="epoch",
        save_total_limit=1,
        fp16=torch.cuda.is_available(),
        dataloader_num_workers=0,
        remove_unused_columns=False,
        report_to="none",
    )

    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    # Trainer
    print("\n" + "=" * 80)
    print("Starting training...")
    print("=" * 80)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        data_collator=data_collator,
    )

    # Train
    trainer.train()

    # Save
    print("\nSaving model...")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    # Metadata
    metadata = {
        "model_name": args.model_name,
        "num_epochs": args.num_epochs,
        "minimal_packages": True,
        "packages": ["torch", "transformers", "peft", "datasets"],
    }
    with open(Path(args.output_dir) / "training_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\n{'=' * 80}")
    print(f"Training complete! Model saved to: {args.output_dir}")
    print(f"{'=' * 80}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--data_dir", type=str, default="training_data")
    parser.add_argument("--output_dir", type=str, default="./qwen-epi-forecast")
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=2e-4)

    args = parser.parse_args()

    if not Path(args.data_dir).exists():
        raise ValueError(f"Data directory not found: {args.data_dir}")

    main(args)
