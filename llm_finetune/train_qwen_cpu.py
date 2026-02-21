"""
Simplified CPU-friendly fine-tuning for Qwen2.5-3B-Instruct on MacBook.
This version uses smaller batch sizes and no quantization for CPU compatibility.

Usage:
    python train_qwen_cpu.py --data_dir training_data --output_dir ./qwen-epi-forecast

Note: Training will be MUCH slower on CPU (hours -> days). Consider using a small subset for testing.
"""

import argparse
import json
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model
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
    formatted = tokenizer.apply_chat_template(
        example["messages"],
        tokenize=False,
        add_generation_prompt=False
    )
    return {"text": formatted}


def main(args):
    print("=" * 80)
    print("Qwen2.5-3B-Instruct CPU Fine-Tuning (MacBook Compatible)")
    print("=" * 80)
    print("\n⚠️  CPU training is SLOW. Consider:")
    print("  - Use --max_samples to train on subset (e.g., --max_samples 100)")
    print("  - Use --num_epochs 1 for quick testing")
    print("  - Use cloud GPU for full training\n")

    # Load dataset
    print("\n[1/5] Loading dataset...")
    dataset = load_jsonl_dataset(args.data_dir)

    # Optionally limit dataset size for faster CPU training
    if args.max_samples:
        print(f"  Limiting to {args.max_samples} training samples for CPU testing...")
        dataset["train"] = dataset["train"].select(range(min(args.max_samples, len(dataset["train"]))))
        val_samples = min(args.max_samples // 10, len(dataset["validation"]))
        dataset["validation"] = dataset["validation"].select(range(val_samples))

    print(f"  Train: {len(dataset['train'])} examples")
    print(f"  Val:   {len(dataset['validation'])} examples")
    print(f"  Test:  {len(dataset['test'])} examples")

    # Load tokenizer
    print("\n[2/5] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        trust_remote_code=True,
        padding_side="right",
    )
    tokenizer.pad_token = tokenizer.eos_token

    # Format dataset
    print("\n[3/5] Formatting dataset with chat template...")
    dataset = dataset.map(
        lambda x: format_chat_template(x, tokenizer),
        remove_columns=dataset["train"].column_names
    )

    # Load model (CPU, no quantization)
    print(f"\n[4/5] Loading model: {args.model_name} (CPU mode)...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        trust_remote_code=True,
        torch_dtype=torch.float32,  # Full precision for CPU
        low_cpu_mem_usage=True,
    )

    # Configure LoRA (smaller for CPU)
    print("\n[5/5] Configuring LoRA...")
    peft_config = LoraConfig(
        r=8,  # Reduced from 16 for faster CPU training
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # Fewer modules
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # Training arguments (CPU-optimized)
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=1,  # Small batch for CPU
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        logging_steps=5,
        save_strategy="epoch",
        eval_strategy="epoch",  # Changed from evaluation_strategy
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        fp16=False,  # No mixed precision on CPU
        dataloader_num_workers=0,  # Important for MacOS stability
        report_to="none",
        seed=42,
    )

    # Initialize trainer
    print("\n" + "=" * 80)
    print("Starting training...")
    print("=" * 80)
    print("\n⏱️  Estimated time for {} samples: {} minutes".format(
        len(dataset["train"]),
        len(dataset["train"]) * args.num_epochs * 2  # Rough estimate: 2 min/sample
    ))
    print("Press Ctrl+C to stop training at any time.\n")

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        processing_class=tokenizer,
    )

    # Train
    try:
        trainer.train()
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user. Saving current model...")

    # Save final model
    print("\nSaving final model...")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    # Save training metadata
    metadata = {
        "model_name": args.model_name,
        "num_epochs": args.num_epochs,
        "batch_size": 1,
        "learning_rate": args.learning_rate,
        "lora_r": 8,
        "lora_alpha": 16,
        "train_samples": len(dataset["train"]),
        "val_samples": len(dataset["validation"]),
        "device": "cpu",
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
        default="./qwen-epi-forecast-cpu",
        help="Output directory for checkpoints"
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=1,
        help="Number of training epochs (default: 1 for CPU)"
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=8,
        help="Gradient accumulation steps (effective batch = 8)"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=2e-4,
        help="Learning rate"
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Limit training to N samples for testing (e.g., 50)"
    )

    args = parser.parse_args()

    # Validate data directory
    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        raise ValueError(f"Data directory not found: {data_dir}")

    main(args)
