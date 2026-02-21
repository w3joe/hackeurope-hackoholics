# Linux CLI Setup Guide

Complete guide for running Qwen2.5-3B fine-tuning on Linux servers with GPU.

## 🖥️ System Requirements

### Minimum
- **OS:** Ubuntu 20.04+ / CentOS 7+ / Debian 10+
- **GPU:** NVIDIA GPU with 12GB+ VRAM (RTX 3090, 4090, A100, etc.)
- **CUDA:** 11.7 or higher
- **RAM:** 32GB+ system RAM
- **Disk:** 50GB free space

### Recommended
- **GPU:** RTX 4090 (24GB) or A100 (40GB/80GB)
- **CUDA:** 12.1+
- **RAM:** 64GB+
- **Disk:** SSD with 100GB+ free space

## 🚀 Quick Start (3 Commands)

```bash
# 1. Setup environment
bash setup_linux.sh

# 2. Run complete training pipeline
bash run_training.sh

# 3. Test the model
source venv/bin/activate
python inference.py --model_path ./qwen-epi-forecast --interactive
```

## 📋 Step-by-Step Setup

### 1. Transfer Files to Linux Server

If you're working remotely, transfer the llm_finetune directory:

```bash
# From your local machine (macOS)
cd /Users/w3joe/Documents/projects/hackeurope-hackoholics
scp -r llm_finetune user@server:/path/to/destination/
```

Or use git:

```bash
# On the Linux server
git clone <your-repo-url>
cd hackeurope-hackoholics/llm_finetune
```

### 2. Verify GPU

Check that CUDA and GPU are available:

```bash
# Check NVIDIA driver
nvidia-smi

# Check CUDA version
nvcc --version
```

Expected output should show your GPU details and CUDA version.

### 3. Run Setup Script

```bash
chmod +x setup_linux.sh
bash setup_linux.sh
```

This will:
- Create a Python virtual environment
- Install PyTorch with CUDA support
- Install all dependencies
- Verify GPU access

### 4. Prepare Training Data

```bash
source venv/bin/activate
python prepare_training_data.py
```

This converts `llm_risk_training.csv` into instruction-tuning format.

### 5. Start Training

#### Option A: Automated Pipeline
```bash
bash run_training.sh
```

#### Option B: Manual Training
```bash
source venv/bin/activate

# Train with custom parameters
python train_qwen.py \
    --data_dir training_data \
    --output_dir ./qwen-epi-forecast \
    --num_epochs 3 \
    --batch_size 4 \
    --gradient_accumulation_steps 4 \
    --learning_rate 2e-4
```

### 6. Monitor Training

Training will show progress like:

```
[1/6] Loading dataset...
  Train: 2543 examples
  Val:   134 examples
  Test:  445 examples

[2/6] Loading tokenizer...
[3/6] Formatting dataset with chat template...
[4/6] Configuring 4-bit quantization...
[5/6] Loading model: Qwen/Qwen2.5-3B-Instruct
[6/6] Configuring LoRA...

trainable params: 5,603,328 || all params: 3,090,603,328 || trainable%: 0.1813

Starting training...
{'loss': 1.234, 'learning_rate': 0.0002, 'epoch': 0.5}
...
```

## 🔧 Advanced Usage

### Running in Background (tmux/screen)

For long training runs, use tmux or screen to keep training running even if disconnected:

```bash
# Using tmux (recommended)
tmux new -s qwen-training
bash run_training.sh
# Press Ctrl+B then D to detach

# Reattach later
tmux attach -t qwen-training
```

```bash
# Using screen
screen -S qwen-training
bash run_training.sh
# Press Ctrl+A then D to detach

# Reattach later
screen -r qwen-training
```

### Using nohup

```bash
nohup bash run_training.sh > training.log 2>&1 &
tail -f training.log  # Monitor progress
```

### Multiple GPUs

If you have multiple GPUs, you can use them all:

```bash
# Automatic multi-GPU (uses all available GPUs)
CUDA_VISIBLE_DEVICES=0,1,2,3 python train_qwen.py \
    --data_dir training_data \
    --output_dir ./qwen-epi-forecast \
    --num_epochs 3 \
    --batch_size 2 \
    --gradient_accumulation_steps 8

# Or use specific GPU
CUDA_VISIBLE_DEVICES=0 python train_qwen.py ...
```

### Adjust Batch Size for Your GPU

| GPU Model | VRAM | Recommended Batch Size | Gradient Accumulation |
|-----------|------|------------------------|----------------------|
| RTX 3090 | 24GB | 4 | 4 |
| RTX 4090 | 24GB | 6 | 3 |
| A100 40GB | 40GB | 8 | 2 |
| A100 80GB | 80GB | 16 | 1 |
| V100 16GB | 16GB | 2 | 8 |

```bash
# For smaller GPU (e.g., RTX 3080 10GB)
python train_qwen.py \
    --batch_size 2 \
    --gradient_accumulation_steps 8

# For larger GPU (e.g., A100 80GB)
python train_qwen.py \
    --batch_size 16 \
    --gradient_accumulation_steps 1
```

## 📊 Monitoring Training

### GPU Utilization

Watch GPU usage in real-time:

```bash
watch -n 1 nvidia-smi
```

### Disk Space

Monitor disk usage during training:

```bash
df -h
du -sh ./qwen-epi-forecast
```

### Training Logs

View training progress:

```bash
# If using run_training.sh with nohup
tail -f training.log

# Check evaluation results
cat ./qwen-epi-forecast/evaluation_results.json | python -m json.tool
```

## 🐛 Troubleshooting

### CUDA Out of Memory

**Solution 1:** Reduce batch size
```bash
python train_qwen.py --batch_size 2 --gradient_accumulation_steps 8
```

**Solution 2:** Reduce max sequence length
Edit `train_qwen.py` line 152:
```python
max_seq_length=1024,  # Changed from 2048
```

### bitsandbytes Not Found

Install from source:
```bash
pip install git+https://github.com/TimDettmers/bitsandbytes.git
```

### Slow Training

**Check GPU usage:**
```bash
nvidia-smi
```

If GPU utilization is low (<80%), try:
- Increase batch size
- Check if CPU is bottleneck (data loading)
- Enable mixed precision (already enabled by default)

### Connection Dropped During Training

Use tmux/screen (see Advanced Usage above) to keep training running.

## 📤 Downloading Trained Model

After training completes, download the model to your local machine:

```bash
# From your local machine
scp -r user@server:/path/to/qwen-epi-forecast ./
```

Or compress first to save bandwidth:

```bash
# On server
tar -czf qwen-epi-forecast.tar.gz qwen-epi-forecast/

# On local machine
scp user@server:/path/to/qwen-epi-forecast.tar.gz ./
tar -xzf qwen-epi-forecast.tar.gz
```

## 🧪 Testing the Model

### Interactive Testing

```bash
python inference.py --model_path ./qwen-epi-forecast --interactive
```

### Batch Testing

```bash
python inference.py \
    --model_path ./qwen-epi-forecast \
    --test_file training_data/test.jsonl \
    --output_file results.json
```

### Evaluation

```bash
python evaluate.py \
    --model_path ./qwen-epi-forecast \
    --test_file training_data/test.jsonl \
    --save_results
```

## 💾 Storage Management

Training generates significant data. Here's what uses space:

```
.
├── llm_risk_training.csv          (~1MB)
├── training_data/                 (~5MB)
├── venv/                          (~2GB - Python packages)
└── qwen-epi-forecast/            (~6-8GB - Model checkpoints)
    ├── checkpoint-xxx/            (~6GB per checkpoint)
    └── evaluation_results.json    (~1MB)
```

**To save space:**

```bash
# Remove intermediate checkpoints (keep only best)
rm -rf qwen-epi-forecast/checkpoint-*

# Remove cache
rm -rf ~/.cache/huggingface/

# Compress final model
tar -czf qwen-epi-forecast.tar.gz qwen-epi-forecast/
```

## 🔐 Security Best Practices

1. **Don't commit model files to git:**
   ```bash
   echo "qwen-epi-forecast/" >> .gitignore
   echo "*.tar.gz" >> .gitignore
   ```

2. **Use SSH keys for remote access**

3. **Monitor GPU processes:**
   ```bash
   fuser -v /dev/nvidia*
   ```

## 📈 Performance Benchmarks

Approximate training times (3 epochs on 2500 examples):

| GPU | Batch Size | Time per Epoch | Total Time |
|-----|-----------|----------------|------------|
| RTX 3090 | 4 | 60 min | 3 hours |
| RTX 4090 | 6 | 40 min | 2 hours |
| A100 40GB | 8 | 30 min | 1.5 hours |
| A100 80GB | 16 | 20 min | 1 hour |

## 🆘 Getting Help

If you encounter issues:

1. Check GPU: `nvidia-smi`
2. Check Python: `python --version` (need 3.8+)
3. Check CUDA: `nvcc --version`
4. Verify packages: `pip list | grep -E "torch|transformers|peft"`
5. Check logs: `tail -f training.log`

## 📚 Additional Resources

- [Hugging Face Transformers Docs](https://huggingface.co/docs/transformers)
- [PEFT Documentation](https://huggingface.co/docs/peft)
- [QLoRA Paper](https://arxiv.org/abs/2305.14314)
- [Qwen2.5 Model Card](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct)

---

**Ready to train?** Run `bash run_training.sh` and you're all set! 🚀
