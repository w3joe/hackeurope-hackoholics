"""
Ultra-fast Qwen2.5-3B training optimized for MacBook CPU.
Multiple speed optimizations for maximum training speed.

Usage:
    # Fastest (10 samples, 2-3 minutes)
    python train_ultra_fast.py --max_samples 10

    # Quick (50 samples, 10-15 minutes)
    python train_ultra_fast.py --max_samples 50

    # Medium (200 samples, 30-45 minutes)
    python train_ultra_fast.py --max_samples 200

Speed optimizations:
✓ LoRA r=4 (minimal rank)
✓ Only 2 attention layers targeted
✓ Max 256 tokens (very short)
✓ No evaluation during training
✓ No checkpointing (saves at end only)
✓ Constant learning rate (no warmup)
✓ Large gradient accumulation
✓ Aggressive truncation
"""

import argparse
import json
from pathlib import Path
import time

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
    }
    return load_dataset("json", data_files=data_files)


def format_chat_template_fast(example, tokenizer, max_length=256):
    """Format and aggressively truncate for speed."""
    formatted = tokenizer.apply_chat_template(
        example["messages"],
        tokenize=False,
        add_generation_prompt=False
    )

    # Truncate to max_length tokens
    tokens = tokenizer(formatted, truncation=True, max_length=max_length)
    formatted_truncated = tokenizer.decode(tokens["input_ids"], skip_special_tokens=True)

    return {"text": formatted_truncated}


def print_speed_tips():
    """Print speed optimization tips."""
    print("\n💡 SPEED TIPS:")
    print("  • Close other apps to free RAM")
    print("  • Disable antivirus temporarily")
    print("  • Use --max_samples 10 for testing")
    print("  • On Apple Silicon (M1/M2/M3): 2-3x faster than Intel")
    print()


def main(args):
    start_time = time.time()

    print("=" * 80)
    print("⚡ ULTRA-FAST Qwen2.5-3B Training (CPU Optimized)")
    print("=" * 80)

    print("\n🚀 ACTIVE SPEED OPTIMIZATIONS:")
    print("  ✓ LoRA rank: 4 (75% fewer params than r=16)")
    print("  ✓ Target modules: 2 layers only (q_proj, v_proj)")
    print("  ✓ Max sequence: 256 tokens (4x faster than 1024)")
    print("  ✓ Gradient accumulation: 32 (very fast updates)")
    print("  ✓ No validation during training")
    print("  ✓ No intermediate checkpoints")
    print("  ✓ Constant LR (no scheduler overhead)")
    print("  ✓ Aggressive text truncation")

    print_speed_tips()

    # Load dataset
    print("[1/5] Loading dataset...")
    dataset = load_jsonl_dataset(args.data_dir)

    # Limit dataset
    if args.max_samples:
        n_train = min(args.max_samples, len(dataset["train"]))
        dataset["train"] = dataset["train"].select(range(n_train))
        print(f"  ✓ Limited to {n_train} training samples")

    print(f"  ✓ Train: {len(dataset['train'])} examples")

    # Load tokenizer
    print("\n[2/5] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        trust_remote_code=True,
        padding_side="right",
    )
    tokenizer.pad_token = tokenizer.eos_token
    print("  ✓ Tokenizer ready")

    # Format dataset with aggressive truncation
    print("\n[3/5] Formatting dataset (truncating to 256 tokens)...")
    dataset = dataset.map(
        lambda x: format_chat_template_fast(x, tokenizer, max_length=256),
        remove_columns=dataset["train"].column_names,
        desc="Formatting"
    )
    print("  ✓ Dataset formatted")

    # Load model
    print(f"\n[4/5] Loading model: {args.model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        trust_remote_code=True,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
    )
    print("  ✓ Base model loaded")

    # Ultra-minimal LoRA config
    print("\n[5/5] Configuring ultra-minimal LoRA...")
    peft_config = LoraConfig(
        r=4,                    # Minimal rank
        lora_alpha=8,           # 2x rank
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "v_proj"],  # Only 2 modules!
    )
    model = get_peft_model(model, peft_config)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())

    print(f"  ✓ Trainable params: {trainable_params:,} ({100 * trainable_params / total_params:.2f}%)")

    # Ultra-fast training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,

        # Training config
        num_train_epochs=args.num_epochs,
        max_steps=args.max_steps if args.max_steps else -1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=32,  # Large = fewer updates = faster

        # Learning rate
        learning_rate=5e-4,              # Higher = faster convergence
        lr_scheduler_type="constant",    # No scheduler = faster
        warmup_steps=0,                  # No warmup = faster

        # Logging & saving
        logging_steps=1,                 # Log every step
        save_strategy="no",              # Don't save during training
        eval_strategy="no",              # No evaluation

        # Optimization
        fp16=False,
        dataloader_num_workers=0,
        optim="adamw_torch",             # Simple optimizer

        # Other
        report_to="none",
        seed=42,
        disable_tqdm=False,
    )

    # Initialize trainer
    print("\n" + "=" * 80)
    print("🚀 STARTING ULTRA-FAST TRAINING")
    print("=" * 80)

    n_samples = len(dataset["train"])
    steps_per_epoch = n_samples // training_args.gradient_accumulation_steps
    total_steps = steps_per_epoch * args.num_epochs

    # Estimate time (based on CPU performance)
    est_seconds_per_step = 3  # Conservative estimate
    est_total_minutes = (total_steps * est_seconds_per_step) / 60

    print(f"\n📊 Training Info:")
    print(f"  Samples: {n_samples}")
    print(f"  Steps per epoch: {steps_per_epoch}")
    print(f"  Total steps: {total_steps}")
    print(f"  ⏱️  Estimated time: {est_total_minutes:.1f} minutes")
    print()

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        processing_class=tokenizer,
    )

    # Train
    try:
        print("Training started... (Press Ctrl+C to stop)\n")
        trainer.train()
        print("\n✓ Training completed!")

    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user")

    # Save final model
    print("\n💾 Saving model...")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    # Save metadata
    elapsed_time = time.time() - start_time
    metadata = {
        "model_name": args.model_name,
        "optimization_level": "ultra_fast",
        "lora_r": 4,
        "lora_alpha": 8,
        "target_modules": ["q_proj", "v_proj"],
        "max_seq_length": 256,
        "train_samples": n_samples,
        "num_epochs": args.num_epochs,
        "training_time_seconds": elapsed_time,
        "training_time_minutes": elapsed_time / 60,
        "device": "cpu",
    }

    with open(Path(args.output_dir) / "training_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\n{'=' * 80}")
    print(f"✅ TRAINING COMPLETE!")
    print(f"{'=' * 80}")
    print(f"Model saved to: {args.output_dir}")
    print(f"⏱️  Total time: {elapsed_time/60:.1f} minutes ({elapsed_time:.0f} seconds)")
    print(f"📈 Speed: {n_samples/(elapsed_time/60):.1f} samples/minute")
    print(f"\nTo test the model:")
    print(f"  python quick_predict.py --model_path {args.output_dir}")
    print(f"{'=' * 80}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Ultra-fast Qwen fine-tuning for CPU"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="Qwen/Qwen2.5-3B-Instruct",
        help="Base model"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="training_data",
        help="Training data directory"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./qwen-ultra-fast",
        help="Output directory"
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=1,
        help="Number of epochs (default: 1)"
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Limit training samples (e.g., 10, 50, 200)"
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=None,
        help="Stop after N steps (overrides epochs)"
    )

    args = parser.parse_args()

    if not Path(args.data_dir).exists():
        raise ValueError(f"Data directory not found: {args.data_dir}")

    main(args)
