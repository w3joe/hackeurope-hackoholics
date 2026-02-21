"""
Ultra-fast CPU training for Qwen2.5-3B-Instruct (optimized for MacBook).
Uses aggressive optimizations for maximum speed on CPU.

Usage:
    python train_qwen_fast.py --max_samples 20 --num_epochs 1

Speed improvements:
- Smaller LoRA rank (r=4 instead of 8)
- Fewer target modules (only attention)
- Shorter max sequence length (512 tokens)
- Larger gradient accumulation (faster updates)
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


def format_chat_template(example, tokenizer, max_length=512):
    """Format messages and truncate for speed."""
    formatted = tokenizer.apply_chat_template(
        example["messages"],
        tokenize=False,
        add_generation_prompt=False
    )
    # Truncate long examples for faster training
    tokens = tokenizer(formatted, truncation=True, max_length=max_length)
    formatted_truncated = tokenizer.decode(tokens["input_ids"], skip_special_tokens=True)
    return {"text": formatted_truncated}


def main(args):
    print("=" * 80)
    print("Qwen2.5-3B-Instruct FAST CPU Training (Speed Optimized)")
    print("=" * 80)
    print("\n⚡ Speed optimizations enabled:")
    print("  - LoRA rank: 4 (reduced from 8)")
    print("  - Target modules: attention only")
    print("  - Max sequence: 512 tokens (reduced from 1024)")
    print("  - Gradient accumulation: 16 (faster updates)")
    print()

    # Load dataset
    print("\n[1/5] Loading dataset...")
    dataset = load_jsonl_dataset(args.data_dir)

    # Limit dataset size
    if args.max_samples:
        print(f"  Limiting to {args.max_samples} training samples...")
        dataset["train"] = dataset["train"].select(range(min(args.max_samples, len(dataset["train"]))))
        val_samples = min(args.max_samples // 10, len(dataset["validation"]))
        dataset["validation"] = dataset["validation"].select(range(val_samples))

    print(f"  Train: {len(dataset['train'])} examples")
    print(f"  Val:   {len(dataset['validation'])} examples")

    # Load tokenizer
    print("\n[2/5] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        trust_remote_code=True,
        padding_side="right",
    )
    tokenizer.pad_token = tokenizer.eos_token

    # Format dataset with truncation
    print("\n[3/5] Formatting dataset (with truncation to 512 tokens)...")
    dataset = dataset.map(
        lambda x: format_chat_template(x, tokenizer, max_length=512),
        remove_columns=dataset["train"].column_names
    )

    # Load model (CPU, no quantization)
    print(f"\n[4/5] Loading model: {args.model_name} (CPU mode)...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        trust_remote_code=True,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
    )

    # Configure LoRA (minimal for speed)
    print("\n[5/5] Configuring minimal LoRA (r=4, attention only)...")
    peft_config = LoraConfig(
        r=4,  # Minimal rank for maximum speed
        lora_alpha=8,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "v_proj"],  # Only attention, skip MLP
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # Training arguments (speed-optimized)
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=16,  # Larger = faster updates
        learning_rate=args.learning_rate,
        lr_scheduler_type="constant",  # Faster than cosine
        warmup_steps=0,  # Skip warmup for speed
        logging_steps=2,  # Log more frequently
        save_strategy="no",  # Don't save checkpoints during training
        eval_strategy="no",  # Skip evaluation for speed
        fp16=False,
        dataloader_num_workers=0,
        report_to="none",
        seed=42,
        max_steps=args.max_steps if args.max_steps else -1,  # Optional early stopping
    )

    # Initialize trainer
    print("\n" + "=" * 80)
    print("Starting FAST training...")
    print("=" * 80)

    estimated_time = len(dataset["train"]) * args.num_epochs * 0.5  # ~30 sec/sample
    print(f"\n⏱️  Estimated time: {estimated_time:.0f} seconds ({estimated_time/60:.1f} minutes)")
    print("Press Ctrl+C to stop training at any time.\n")

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
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

    # Save metadata
    metadata = {
        "model_name": args.model_name,
        "num_epochs": args.num_epochs,
        "batch_size": 1,
        "learning_rate": args.learning_rate,
        "lora_r": 4,
        "lora_alpha": 8,
        "train_samples": len(dataset["train"]),
        "device": "cpu",
        "optimization": "fast",
        "max_seq_length": 512,
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
        default="./qwen-fast",
        help="Output directory for checkpoints"
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=1,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=3e-4,
        help="Learning rate (higher for faster convergence)"
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=20,
        help="Limit training to N samples (default: 20 for fast testing)"
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=None,
        help="Stop after N steps (overrides epochs)"
    )

    args = parser.parse_args()

    # Validate data directory
    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        raise ValueError(f"Data directory not found: {data_dir}")

    main(args)
