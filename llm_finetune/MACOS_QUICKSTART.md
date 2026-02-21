# MacOS Quick Start Guide

Simple guide to fine-tune Qwen2.5-3B on your MacBook (CPU only).

## ⚠️ Important Notes

- **CPU training is SLOW**: 100x slower than GPU
- **Recommended for testing only**: Train on small subset first
- **For production**: Use Linux with GPU or cloud services (Google Colab, AWS, etc.)

## 🚀 Quick Start (5 Minutes Setup)

### 1. Install Dependencies

```bash
cd /Users/w3joe/Documents/projects/hackeurope-hackoholics/llm_finetune

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install packages (MacOS compatible, no GPU dependencies)
pip install -r requirements_macos.txt
```

### 2. Prepare Training Data

```bash
python prepare_training_data.py
```

This creates:
- `training_data/train.jsonl` (~2500 examples)
- `training_data/val.jsonl` (~130 examples)
- `training_data/test.jsonl` (~440 examples)

### 3. Quick Test Training (50 samples, ~30 minutes)

```bash
python train_qwen_cpu.py \
    --data_dir training_data \
    --output_dir ./qwen-test \
    --max_samples 50 \
    --num_epochs 1
```

### 4. Full Training (All samples, 2-3 days!)

```bash
# Only run this if you're patient!
python train_qwen_cpu.py \
    --data_dir training_data \
    --output_dir ./qwen-epi-forecast-cpu \
    --num_epochs 1
```

### 5. Test the Model

```bash
python inference.py \
    --model_path ./qwen-test \
    --interactive
```

## 💡 Recommended Workflow

### For Testing on MacBook

1. **Quick validation** (10-50 samples):
   ```bash
   python train_qwen_cpu.py --max_samples 50 --num_epochs 1
   ```
   - Time: ~30-60 minutes
   - Purpose: Verify code works, see if model learns

2. **Local debugging** (100-200 samples):
   ```bash
   python train_qwen_cpu.py --max_samples 200 --num_epochs 1
   ```
   - Time: 2-4 hours
   - Purpose: Test different hyperparameters

### For Production Training

**Option 1: Google Colab (Free GPU)**
1. Upload files to Google Drive
2. Open Colab notebook
3. Use the GPU training script (`train_qwen.py`)
4. Download trained model

**Option 2: Cloud GPU (AWS/GCP/Azure)**
1. Copy files to cloud instance
2. Follow [LINUX_SETUP.md](LINUX_SETUP.md)
3. Train in 2-4 hours

## 🎯 Training Options

| Mode | Samples | Epochs | Time | Use Case |
|------|---------|--------|------|----------|
| Quick Test | 50 | 1 | 30 min | Verify code works |
| Debug | 200 | 1 | 2-4 hours | Test hyperparameters |
| Small | 500 | 1 | 8-12 hours | Limited training |
| Full CPU | All | 1 | 2-3 days | Patience required |
| **GPU (Recommended)** | All | 3 | 2-4 hours | Production ready |

## 📝 Example Commands

### Minimal Test (Fastest)
```bash
python train_qwen_cpu.py \
    --max_samples 20 \
    --num_epochs 1 \
    --output_dir ./test-model
```

### Medium Test
```bash
python train_qwen_cpu.py \
    --max_samples 100 \
    --num_epochs 2 \
    --output_dir ./medium-model
```

### Custom Configuration
```bash
python train_qwen_cpu.py \
    --data_dir training_data \
    --output_dir ./custom-model \
    --max_samples 200 \
    --num_epochs 1 \
    --learning_rate 1e-4 \
    --gradient_accumulation_steps 16
```

## 🔍 Monitor Training

Training will show progress:

```
[1/5] Loading dataset...
  Limiting to 50 training samples for CPU testing...
  Train: 50 examples
  Val:   5 examples

[2/5] Loading tokenizer...
[3/5] Formatting dataset...
[4/5] Loading model (CPU mode)...
[5/5] Configuring LoRA...

trainable params: 2,359,296 || all params: 3,087,359,296 || trainable%: 0.0764

Starting training...
⏱️  Estimated time for 50 samples: 100 minutes

{'loss': 2.134, 'learning_rate': 0.0002, 'epoch': 0.1}
{'loss': 1.856, 'learning_rate': 0.00018, 'epoch': 0.2}
...
```

## 🛑 Stop Training Anytime

Press `Ctrl+C` to stop training. The current model will be saved automatically.

## 💻 System Requirements

- **RAM**: 16GB+ recommended (8GB minimum)
- **Disk**: 20GB free space
- **CPU**: Apple Silicon (M1/M2/M3) recommended, Intel works but slower
- **Time**: Patience! ☕

## 🚀 After Training

### Test Inference
```bash
python inference.py --model_path ./qwen-test --interactive
```

### Evaluate
```bash
python evaluate.py \
    --model_path ./qwen-test \
    --test_file training_data/test.jsonl
```

## 🐛 Troubleshooting

### "Killed" or Memory Error
- Reduce `--max_samples` (try 20-50)
- Close other applications
- Restart terminal and try again

### Very Slow Training
- This is normal on CPU!
- Use `--max_samples 50` for testing
- Consider cloud GPU for full training

### Model Not Loading
```bash
# Verify installation
python -c "import torch, transformers, peft; print('OK')"

# Check model downloaded
ls -lh ~/.cache/huggingface/hub/
```

## 📊 Performance Comparison

Training 100 samples for 1 epoch:

| Device | Time | Cost |
|--------|------|------|
| MacBook Pro M1 (CPU) | ~2 hours | Free |
| MacBook Pro M2 (CPU) | ~1.5 hours | Free |
| Google Colab (T4 GPU) | ~5 minutes | Free |
| RTX 4090 (GPU) | ~2 minutes | $$ |

## 🌟 Next Steps

1. **Quick test**: Run with `--max_samples 50`
2. **Verify it works**: Check inference works
3. **Move to GPU**: Use cloud for full training
4. **Deploy**: Use trained model for predictions

## 📚 Files You Need

- ✅ `llm_risk_training.csv` - Your training data
- ✅ `prepare_training_data.py` - Data preprocessing
- ✅ `train_qwen_cpu.py` - CPU training script (this)
- ✅ `inference.py` - Run predictions
- ✅ `evaluate.py` - Evaluate model
- ✅ `requirements_macos.txt` - Dependencies

## 💡 Pro Tips

1. **Start small**: Always test with `--max_samples 50` first
2. **Monitor memory**: Use Activity Monitor to watch RAM usage
3. **Save checkpoints**: Training saves at each epoch
4. **Use tmux/screen**: For long training runs
5. **Consider cloud**: If training > 4 hours, use GPU instead

---

**Ready to train?** Start with the Quick Test above! 🎉
