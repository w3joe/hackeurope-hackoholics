# Storage Emergency Guide - University Quota

You're at 85% capacity on `/cs/student/ug`. Here's how to fix it immediately.

## 🚨 Immediate Actions

### 1. Find What's Using Space (Run This First)

```bash
# Check overall usage
df -h

# Find largest directories
du -sh ~/* | sort -hr | head -20

# Find largest files
find ~ -type f -size +100M 2>/dev/null -exec ls -lh {} \; | awk '{print $9, $5}'
```

### 2. Common Culprits (Usually These)

#### A. Hugging Face Cache (~5-10GB!)
```bash
# Check size
du -sh ~/.cache/huggingface

# Clear it (models will re-download when needed)
rm -rf ~/.cache/huggingface/hub/*
```

#### B. Pip Cache (~1-3GB)
```bash
# Check size
du -sh ~/.cache/pip

# Clear it
pip cache purge
```

#### C. Virtual Environments (~2-5GB each)
```bash
# Find all venvs
find ~ -maxdepth 3 -name "venv" -o -name ".venv"

# Delete old ones you don't need
rm -rf ~/old_project/venv
```

#### D. PyTorch Cache
```bash
du -sh ~/.cache/torch
rm -rf ~/.cache/torch/*
```

#### E. Node Modules (if you have them)
```bash
find ~ -name "node_modules" -type d
# Delete any you don't need
```

---

## 🎯 Quick Cleanup Script

```bash
# Download and run
cd /cs/student/ug/your_username
chmod +x cleanup_storage.sh
bash cleanup_storage.sh
```

Or manually:

```bash
# Clear all caches at once (SAFE - will re-download when needed)
pip cache purge
rm -rf ~/.cache/huggingface/hub/*
rm -rf ~/.cache/torch/*
rm -rf ~/.cache/pip/*
```

**This usually frees 5-15GB!**

---

## 💾 For Your LLM Training Project

### Option 1: Don't Store Models Locally

Use the **streaming script** I created:

```bash
python train_qwen_streaming.py \
    --data_dir training_data \
    --output_dir ./qwen-epi-forecast \
    --num_epochs 3 \
    --batch_size 8 \
    --temp_cache \
    --clean_cache
```

**This:**
- ✅ Streams model from HF Hub (no local storage)
- ✅ Uses `/tmp` for cache (auto-deleted)
- ✅ Only saves 15MB adapter
- ✅ Cleans up after training

### Option 2: Use Temporary Storage

```bash
# Use /tmp (cleared on reboot)
export HF_HOME=/tmp/huggingface
export TRANSFORMERS_CACHE=/tmp/transformers

# Then train
python train_qwen_streaming.py ...
```

### Option 3: Use Google Colab (BEST)

Since you're out of space, **use Google Colab**:
- ✅ FREE GPU
- ✅ No storage quota issues
- ✅ 100GB+ free space
- ✅ Pre-installed packages

---

## 📊 Storage Budget for Training

| Item | Size | Can Delete? |
|------|------|-------------|
| Base model download | ~6GB | ✅ Yes (use streaming) |
| Hugging Face cache | ~5-10GB | ✅ Yes (clears automatically) |
| Virtual environment | ~3-5GB | ❌ Need for training |
| Training data | ~5MB | ❌ Need for training |
| LoRA adapter (result) | ~15MB | ❌ Keep this! |
| Pip cache | ~1-3GB | ✅ Yes (clears automatically) |

**Minimum needed with streaming:** ~3-5GB (just venv)

---

## 🔧 Emergency Cleanup Commands

### Nuclear Option (Clear Everything)
```bash
# WARNING: This clears ALL caches
rm -rf ~/.cache/*

# Re-check space
df -h
```

### Delete Specific Large Files
```bash
# Find files over 1GB
find ~ -type f -size +1G 2>/dev/null

# Delete specific file
rm /path/to/large/file
```

### Clean Old Project Files
```bash
# Find old Python bytecode
find ~ -name "*.pyc" -delete
find ~ -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null

# Find old logs
find ~ -name "*.log" -size +100M -delete
```

---

## 🎯 Recommended Solution for Your Situation

### Immediate (Free 5-10GB in 30 seconds):
```bash
pip cache purge
rm -rf ~/.cache/huggingface/hub/*
rm -rf ~/.cache/torch/*
```

### For Training (Use Streaming):
```bash
python train_qwen_streaming.py \
    --data_dir training_data \
    --output_dir ./qwen-epi-forecast \
    --num_epochs 3 \
    --batch_size 8 \
    --temp_cache \
    --clean_cache
```

### Long-term (Use Google Colab):
No storage issues, free GPU, done!

---

## 📈 Before/After Storage

**Before cleanup:**
```
Filesystem                         Size  Used Avail Use% Mounted on
evs2:/cs/student/ug                4.0G  3.4G  647M  85% /cs/student/ug
```

**After cleanup (expected):**
```
Filesystem                         Size  Used Avail Use% Mounted on
evs2:/cs/student/ug                4.0G  1.5G  2.5G  38% /cs/student/ug
```

---

## 🚀 Action Plan

1. **Right now:** Run cleanup commands above (30 seconds)
2. **For training:** Use streaming script (no model download)
3. **Alternative:** Use Google Colab (no storage issues)

**Want me to help you set up Google Colab instead?** That's the easiest solution! 🎯
