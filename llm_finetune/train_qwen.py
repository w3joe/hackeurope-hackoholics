"""
Fine-tune Qwen2.5-3B-Instruct for epidemiological forecasting using QLoRA.

Usage:
    python train_qwen.py --data_dir training_data --output_dir ./qwen-epi-forecast

Requirements:
    - GPU with 12GB+ VRAM (tested on RTX 3090, 4090)
    - CUDA installed
"""

import argparse
import json
import os
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer


def load_jsonl_dataset(data_dir):
    """Load train/val/test JSONL files."""
    data_files = {
        "train": str(Path(data_dir) / "train.jsonl"),
        "validation": str(Path(data_dir) / "val.jsonl"),
        "test": str(Path(data_dir) / "test.jsonl"),
    }
    return load_dataset("json", data_files=data_files)


def format_chat_template(example, tokenizer):
    """Format messages using Qwen chat template."""
    # Apply the chat template
    formatted = tokenizer.apply_chat_template(
        example["messages"],
        tokenize=False,
        add_generation_prompt=False
    )
    return {"text": formatted}


def main(args):
    print("=" * 80)
    print("Qwen2.5-3B-Instruct Fine-Tuning for Epidemiological Forecasting")
    print("=" * 80)

    # 1. Load dataset
    print("\n[1/6] Loading dataset...")
    dataset = load_jsonl_dataset(args.data_dir)
    print(f"  Train: {len(dataset['train'])} examples")
    print(f"  Val:   {len(dataset['validation'])} examples")
    print(f"  Test:  {len(dataset['test'])} examples")

    # 2. Load tokenizer
    print("\n[2/6] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        trust_remote_code=True,
        padding_side="right",
    )
    tokenizer.pad_token = tokenizer.eos_token

    # 3. Format dataset
    print("\n[3/6] Formatting dataset with chat template...")
    dataset = dataset.map(
        lambda x: format_chat_template(x, tokenizer),
        remove_columns=dataset["train"].column_names
    )

    # 4. Configure quantization (QLoRA)
    print("\n[4/6] Configuring 4-bit quantization...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    # 5. Load model
    print(f"\n[5/6] Loading model: {args.model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.float16,
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model)

    # 6. Configure LoRA
    print("\n[6/6] Configuring LoRA...")
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ],
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
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        logging_steps=10,
        save_strategy="epoch",
        evaluation_strategy="epoch",
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        bf16=True if torch.cuda.is_bf16_supported() else False,
        fp16=not torch.cuda.is_bf16_supported(),
        gradient_checkpointing=True,
        optim="paged_adamw_8bit",
        report_to="none",  # or "wandb" if you want logging
        seed=42,
    )

    # Initialize trainer
    print("\n" + "=" * 80)
    print("Starting training...")
    print("=" * 80)

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        tokenizer=tokenizer,
        dataset_text_field="text",
        max_seq_length=2048,
        packing=False,
    )

    # Train
    trainer.train()

    # Save final model
    print("\nSaving final model...")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    # Save training metadata
    metadata = {
        "model_name": args.model_name,
        "num_epochs": args.num_epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "lora_r": 16,
        "lora_alpha": 32,
        "train_samples": len(dataset["train"]),
        "val_samples": len(dataset["validation"]),
    }
    with open(Path(args.output_dir) / "training_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\n{'=' * 80}")
    print(f"Training complete! Model saved to: {args.output_dir}")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        type=str,
        default="Qwen/Qwen2.5-3B-Instruct",
        help="Base model to fine-tune"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="training_data",
        help="Directory containing train/val/test.jsonl"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./qwen-epi-forecast",
        help="Output directory for checkpoints"
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=3,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Per-device batch size"
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=4,
        help="Gradient accumulation steps (effective batch = batch_size * this)"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=2e-4,
        help="Learning rate"
    )

    args = parser.parse_args()

    # Validate data directory
    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        raise ValueError(f"Data directory not found: {data_dir}")

    main(args)
