# Step-by-Step Installation (Avoiding Storage Issues)

If you're maxing out on package sizes, follow this carefully.

## 🎯 Strategy: Install Smallest Possible Set

### Option 1: Minimal Training (4 Packages Only)

Install **ONLY** these 4 packages:

```bash
# Step 1: Clear everything first
pip cache purge
rm -rf ~/.cache/pip
rm -rf ~/.cache/huggingface

# Step 2: Install PyTorch (smallest version)
pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cu118

# Step 3: Install transformers (without extras)
pip install --no-cache-dir transformers --no-deps

# Step 4: Install required deps for transformers
pip install --no-cache-dir numpy packaging regex requests filelock tokenizers huggingface_hub pyyaml

# Step 5: Install peft
pip install --no-cache-dir peft

# Step 6: Install datasets
pip install --no-cache-dir datasets
```

**Total size:** ~2.5-3GB (vs 5-6GB for full install)

Then use the minimal script:
```bash
python train_minimal_packages.py \
    --data_dir training_data \
    --output_dir ./qwen-epi-forecast \
    --num_epochs 3 \
    --batch_size 2
```

---

### Option 2: One-at-a-Time Installation

If even that's too much, install one at a time and check space:

```bash
# Check space before each step
df -h

# 1. PyTorch (~2GB)
pip install --no-cache-dir torch
df -h

# 2. Transformers (~300MB)
pip install --no-cache-dir transformers
df -h

# 3. PEFT (~50MB)
pip install --no-cache-dir peft
df -h

# 4. Datasets (~200MB)
pip install --no-cache-dir datasets
df -h
```

If you run out of space at any point, **STOP** and use Google Colab instead.

---

### Option 3: CPU-Only PyTorch (Smaller)

If you're training on CPU, use CPU-only PyTorch (saves ~500MB):

```bash
pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
pip install --no-cache-dir transformers peft datasets
```

Then use:
```bash
python train_minimal_packages.py \
    --data_dir training_data \
    --output_dir ./qwen-epi-forecast \
    --num_epochs 2 \
    --batch_size 1
```

---

## 💾 Package Size Breakdown

| Package | Size | Needed For | Can Skip? |
|---------|------|-----------|-----------|
| **torch** | ~2GB | Everything | ❌ Required |
| **transformers** | ~300MB | Everything | ❌ Required |
| **peft** | ~50MB | LoRA training | ❌ Required |
| **datasets** | ~200MB | Data loading | ❌ Required |
| trl | ~30MB | SFTTrainer | ✅ Use Trainer instead |
| accelerate | ~100MB | Multi-GPU | ✅ Single GPU only |
| bitsandbytes | ~50MB | Quantization | ✅ Use fp16 instead |
| numpy | ~50MB | Arrays | ⚠️ transformers needs it |
| pandas | ~50MB | Data prep | ✅ Already have data |
| scikit-learn | ~100MB | Evaluation | ✅ Do later |

**Absolute minimum:** torch + transformers + peft + datasets = ~2.5GB

---

## 🚨 If You Still Run Out of Space

### Solution 1: Use Temporary Storage

```bash
# Create temp directory
mkdir -p /tmp/pip_packages

# Install to temp location
pip install --target=/tmp/pip_packages --no-cache-dir torch transformers peft datasets

# Add to Python path
export PYTHONPATH=/tmp/pip_packages:$PYTHONPATH

# Train (packages in /tmp, auto-deleted on reboot)
python train_minimal_packages.py ...
```

### Solution 2: Use Virtual Environment in /tmp

```bash
# Create venv in /tmp (auto-deleted on reboot)
python3 -m venv /tmp/train_venv
source /tmp/train_venv/bin/activate

# Install packages
pip install --no-cache-dir torch transformers peft datasets

# Train
python train_minimal_packages.py ...

# After training, copy model out of /tmp!
cp -r ./qwen-epi-forecast ~/safe_location/
```

### Solution 3: Google Colab (BEST for Storage Issues)

Seriously, if you keep hitting storage limits, **use Google Colab**:

```python
# In Colab notebook
!pip install peft  # torch & transformers pre-installed

# Upload your training_data to Drive, then:
!python train_minimal_packages.py \
    --data_dir /content/drive/MyDrive/training_data \
    --output_dir /content/drive/MyDrive/qwen-epi-forecast
```

**No storage issues, FREE GPU, done!**

---

## 📊 Monitoring Installation Size

Watch disk usage in real-time:

```bash
# Terminal 1: Run this
watch -n 1 'df -h | grep student'

# Terminal 2: Install packages
pip install --no-cache-dir torch
# (watch Terminal 1 to see space decrease)
```

If space drops below 500MB free, **STOP** and switch to Colab.

---

## ✅ Verification

After installation, verify it works:

```python
python3 << EOF
import torch
import transformers
import peft
import datasets
print("✓ All packages loaded successfully!")
print(f"PyTorch: {torch.__version__}")
print(f"Transformers: {transformers.__version__}")
print(f"PEFT: {peft.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
EOF
```

---

## 🎯 Final Recommendation

**Based on your situation (10GB quota, packages maxing out):**

### If you have 4GB+ free:
```bash
pip install --no-cache-dir torch transformers peft datasets
python train_minimal_packages.py --data_dir training_data --output_dir ./qwen-epi-forecast
```

### If you have less than 4GB free:
**Use Google Colab!** You're fighting a losing battle with storage limits. Colab gives you:
- ✅ 100GB+ space
- ✅ FREE GPU
- ✅ Pre-installed packages
- ✅ No quota headaches

Want me to create a Colab notebook? It's 5 minutes of setup vs hours of fighting storage limits! 🎯
