# Minimal Installation Guide (Low Disk/Quota)

If you're hitting storage quota limits, use this minimal installation approach.

## 🎯 Strategy: Install Only What You Need

Instead of installing everything at once, install packages as needed.

---

## Option 1: Absolute Minimum (For Training Only)

### Step 1: Install Core Only
```bash
pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Step 2: Install Training Dependencies
```bash
pip install --no-cache-dir transformers peft trl accelerate
```

### Step 3: Install Data Loading
```bash
pip install --no-cache-dir datasets
```

### Step 4: GPU Quantization (GPU only)
```bash
pip install --no-cache-dir bitsandbytes
```

**Total packages:** Only 6 (vs 13 in full requirements)

---

## Option 2: Use Pre-installed Packages

If you're on Google Colab, Kaggle, or cloud GPU:

```bash
# These are already installed:
# - torch
# - transformers
# - accelerate

# Install only missing ones:
pip install --no-cache-dir peft trl datasets bitsandbytes
```

---

## Option 3: Skip Heavy Dependencies

### Install Without Optional Features
```bash
# Skip extras
pip install --no-cache-dir transformers --no-deps
pip install --no-cache-dir peft --no-deps
pip install --no-cache-dir trl --no-deps

# Then install only core dependencies manually
pip install --no-cache-dir torch datasets accelerate
```

---

## Option 4: Use System Python (No venv)

If quota is very tight, skip virtual environment:

```bash
# Install directly to user directory
pip install --user --no-cache-dir torch transformers peft trl accelerate datasets bitsandbytes
```

**Saves:** ~500MB (no venv overhead)

---

## Option 5: Cloud-Based Training (No Local Install)

### Google Colab (FREE GPU, pre-installed packages)

1. Upload your files to Google Drive
2. Open Google Colab notebook
3. Install only missing packages:

```python
!pip install peft trl
```

4. Run training:
```python
!python train_qwen_streaming.py \
    --data_dir /content/drive/MyDrive/llm_finetune/training_data \
    --output_dir /content/drive/MyDrive/qwen-epi-forecast \
    --num_epochs 3 \
    --batch_size 8 \
    --temp_cache --clean_cache
```

**Benefits:**
- ✅ No local storage needed
- ✅ Free GPU (T4)
- ✅ Most packages pre-installed
- ✅ Results saved to Google Drive

---

## Option 6: Minimal Script (No Training File Changes)

If you can't modify scripts, create a wrapper:

**`train_minimal.sh`:**
```bash
#!/bin/bash
# Minimal training script - only installs what's needed

echo "Installing minimal dependencies..."
pip install --no-cache-dir --quiet torch transformers peft trl accelerate datasets bitsandbytes

echo "Starting training..."
python train_qwen_streaming.py \
    --data_dir training_data \
    --output_dir ./qwen-epi-forecast \
    --num_epochs 3 \
    --batch_size 8 \
    --temp_cache --clean_cache

echo "Cleaning up..."
pip uninstall -y bitsandbytes  # Free up space after training
```

---

## 💾 Storage Comparison

| Method | Initial Install | During Training | After Training |
|--------|----------------|-----------------|----------------|
| Full requirements.txt | ~5GB | ~8GB | ~6GB |
| Minimal (Option 1) | ~3GB | ~6GB | ~3GB |
| Google Colab | 0GB | 0GB | ~15MB |
| Streaming + temp_cache | ~3GB | ~3GB | ~15MB |

---

## 🚀 Quick Start (Google Colab - RECOMMENDED)

Since you have storage issues, use **Google Colab** (free):

### 1. Create Colab Notebook

```python
# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Install missing packages
!pip install peft trl

# Run training
!python /content/drive/MyDrive/llm_finetune/train_qwen_streaming.py \
    --data_dir /content/drive/MyDrive/llm_finetune/training_data \
    --output_dir /content/drive/MyDrive/qwen-epi-forecast \
    --num_epochs 3 \
    --batch_size 8 \
    --gradient_accumulation_steps 2 \
    --learning_rate 2e-4 \
    --temp_cache \
    --clean_cache
```

### 2. Upload Files to Google Drive
```
Google Drive/
└── llm_finetune/
    ├── train_qwen_streaming.py
    ├── training_data/
    │   ├── train.jsonl
    │   ├── val.jsonl
    │   └── test.jsonl
    └── (results will be saved here)
```

**Benefits:**
- ✅ FREE T4 GPU
- ✅ No local storage needed
- ✅ ~2-3 hour training time
- ✅ Save results to your Drive

---

## 🔧 Troubleshooting Storage Issues

### If you still hit quota during pip install:

#### 1. Clear pip cache
```bash
pip cache purge
```

#### 2. Use `--no-cache-dir` always
```bash
pip install --no-cache-dir <package>
```

#### 3. Install one package at a time
```bash
pip install --no-cache-dir torch
pip install --no-cache-dir transformers
pip install --no-cache-dir peft
# etc...
```

#### 4. Check disk usage
```bash
du -sh ~/.cache/pip
du -sh ~/.cache/huggingface
du -sh venv/
```

#### 5. Clean Hugging Face cache
```bash
rm -rf ~/.cache/huggingface/hub/*
```

---

## ⚡ Absolute Minimum Command (4 packages)

If you can ONLY install 4 packages:

```bash
pip install --no-cache-dir torch transformers peft datasets
```

Then use **train_qwen_cpu.py** (no TRL/accelerate/bitsandbytes needed):

```bash
python train_qwen_cpu.py \
    --max_samples 200 \
    --num_epochs 2 \
    --output_dir ./qwen-minimal
```

---

## 📋 Package Sizes (for reference)

| Package | Size | Required For |
|---------|------|-------------|
| torch | ~2GB | Everything |
| transformers | ~500MB | Everything |
| peft | ~50MB | LoRA training |
| trl | ~30MB | SFTTrainer |
| accelerate | ~100MB | Multi-GPU |
| bitsandbytes | ~50MB | GPU quantization |
| datasets | ~200MB | Data loading |
| pandas | ~50MB | Data prep only |
| scikit-learn | ~100MB | Evaluation only |

**Minimum for GPU training:** ~2.7GB (torch + transformers + peft + trl + accelerate + bitsandbytes + datasets)

**Minimum for CPU training:** ~2.8GB (torch + transformers + peft + datasets)

---

## 🎯 RECOMMENDED SOLUTION

Since you have storage quota issues:

### Use Google Colab (FREE, no storage needed):

1. Upload `llm_finetune` folder to Google Drive
2. Open new Colab notebook
3. Run training (see Quick Start above)
4. Download trained model (~15MB) when done

**This is the easiest solution if you're hitting storage limits!** 🚀

---

Need help setting up Google Colab? Let me know!
