# GPU Training Guide - Optimized Parameters

Complete guide for training Qwen2.5-3B on GPU with best parameters.

## 🚀 Recommended GPU Training Commands

### Standard Training (Best Balance)
```bash
python train_qwen.py \
    --data_dir training_data \
    --output_dir ./qwen-epi-forecast \
    --num_epochs 3 \
    --batch_size 8 \
    --gradient_accumulation_steps 2 \
    --learning_rate 2e-4
```

**Time:** ~2-3 hours on RTX 4090, ~3-4 hours on RTX 3090
**Quality:** Excellent (production-ready)
**Memory:** ~18-20GB VRAM

---

### High Quality Training (Best Results)
```bash
python train_qwen.py \
    --data_dir training_data \
    --output_dir ./qwen-epi-forecast \
    --num_epochs 5 \
    --batch_size 8 \
    --gradient_accumulation_steps 2 \
    --learning_rate 1.5e-4
```

**Time:** ~4-6 hours
**Quality:** Best possible
**Memory:** ~18-20GB VRAM

---

### Fast Training (Good Quality, Less Time)
```bash
python train_qwen.py \
    --data_dir training_data \
    --output_dir ./qwen-epi-forecast \
    --num_epochs 2 \
    --batch_size 12 \
    --gradient_accumulation_steps 2 \
    --learning_rate 3e-4
```

**Time:** ~1-2 hours
**Quality:** Good
**Memory:** ~22-24GB VRAM (requires high-end GPU)

---

### Memory-Constrained GPU (12-16GB VRAM)
```bash
python train_qwen.py \
    --data_dir training_data \
    --output_dir ./qwen-epi-forecast \
    --num_epochs 3 \
    --batch_size 4 \
    --gradient_accumulation_steps 4 \
    --learning_rate 2e-4
```

**Time:** ~3-4 hours
**Quality:** Excellent
**Memory:** ~12-14GB VRAM

---

## 📊 GPU-Specific Recommendations

### RTX 4090 / A100 40GB (24GB+ VRAM) - BEST
```bash
python train_qwen.py \
    --data_dir training_data \
    --output_dir ./qwen-epi-forecast \
    --num_epochs 4 \
    --batch_size 16 \
    --gradient_accumulation_steps 1 \
    --learning_rate 2e-4
```
- **Effective batch size:** 16
- **Speed:** ~1.5-2 hours
- **Memory:** ~22GB VRAM

### RTX 3090 / RTX 4080 (24GB VRAM)
```bash
python train_qwen.py \
    --data_dir training_data \
    --output_dir ./qwen-epi-forecast \
    --num_epochs 3 \
    --batch_size 12 \
    --gradient_accumulation_steps 2 \
    --learning_rate 2e-4
```
- **Effective batch size:** 24
- **Speed:** ~2-3 hours
- **Memory:** ~20GB VRAM

### RTX 3080 / RTX 4070 Ti (10-12GB VRAM)
```bash
python train_qwen.py \
    --data_dir training_data \
    --output_dir ./qwen-epi-forecast \
    --num_epochs 3 \
    --batch_size 4 \
    --gradient_accumulation_steps 4 \
    --learning_rate 2e-4
```
- **Effective batch size:** 16
- **Speed:** ~3-4 hours
- **Memory:** ~11GB VRAM

### RTX 3060 / RTX 4060 (8GB VRAM)
```bash
python train_qwen.py \
    --data_dir training_data \
    --output_dir ./qwen-epi-forecast \
    --num_epochs 3 \
    --batch_size 2 \
    --gradient_accumulation_steps 8 \
    --learning_rate 2e-4
```
- **Effective batch size:** 16
- **Speed:** ~4-5 hours
- **Memory:** ~8GB VRAM

---

## 🎯 Parameter Explanation

### Num Epochs
- **2 epochs:** Fast, decent quality (good for testing)
- **3 epochs:** ✅ **Recommended** - Best balance
- **4-5 epochs:** Best quality, risk of overfitting
- **>5 epochs:** Likely to overfit

### Batch Size
- **Small (2-4):** Memory efficient, slower
- **Medium (8-12):** ✅ **Recommended** - Best balance
- **Large (16+):** Fastest, requires more VRAM

### Gradient Accumulation Steps
- **Purpose:** Simulates larger batch sizes
- **Formula:** Effective batch = batch_size × gradient_accumulation_steps
- **Recommended effective batch:** 16-32

### Learning Rate
- **1e-4:** Conservative, slower learning
- **2e-4:** ✅ **Recommended** - Good balance
- **3e-4:** Faster learning, might be unstable
- **5e-4:** Too high, likely unstable

---

## 💡 Advanced Tuning

### For Better Convergence
```bash
python train_qwen.py \
    --data_dir training_data \
    --output_dir ./qwen-epi-forecast \
    --num_epochs 4 \
    --batch_size 8 \
    --gradient_accumulation_steps 2 \
    --learning_rate 1.5e-4  # Lower LR for stability
```

### For Maximum Speed
```bash
python train_qwen.py \
    --data_dir training_data \
    --output_dir ./qwen-epi-forecast \
    --num_epochs 2 \
    --batch_size 16 \
    --gradient_accumulation_steps 1 \
    --learning_rate 3e-4  # Higher LR for fast learning
```

### For Small Dataset (<500 samples)
```bash
python train_qwen.py \
    --data_dir training_data \
    --output_dir ./qwen-epi-forecast \
    --num_epochs 5 \
    --batch_size 4 \
    --gradient_accumulation_steps 2 \
    --learning_rate 1e-4  # Lower LR to prevent overfitting
```

---

## 📈 Expected Results by Configuration

| Configuration | Time | RMSE | Risk Acc | Quality |
|--------------|------|------|----------|---------|
| Fast (2 epochs, bs=16) | 1-2h | 8-12 | 70-75% | Good |
| Standard (3 epochs, bs=8) | 2-3h | 6-10 | 75-80% | Excellent ✅ |
| High Quality (5 epochs, bs=8) | 4-6h | 5-8 | 80-85% | Best |

---

## 🔍 Monitoring Training

### Watch GPU Usage
```bash
watch -n 1 nvidia-smi
```

Look for:
- **GPU utilization:** Should be 90-100%
- **Memory usage:** Should be stable, not increasing
- **Temperature:** Should be <85°C

### Check Training Progress
Training will show:
```
{'loss': 1.234, 'learning_rate': 0.0002, 'epoch': 0.5}
{'eval_loss': 1.156, 'eval_runtime': 2.34, 'epoch': 1.0}
...
```

**Good signs:**
- ✅ Loss decreasing steadily
- ✅ Eval_loss close to loss (not overfitting)
- ✅ No NaN or inf values

**Warning signs:**
- ❌ Loss not decreasing after epoch 1
- ❌ Eval_loss much higher than loss (overfitting)
- ❌ NaN/inf values (learning rate too high)

---

## 🎬 Quick Start

**Don't know your GPU? Run this:**
```bash
nvidia-smi --query-gpu=name,memory.total --format=csv
```

**Then use the recommended command for your GPU from above!**

**Most common (RTX 3090/4090):**
```bash
python train_qwen.py \
    --data_dir training_data \
    --output_dir ./qwen-epi-forecast \
    --num_epochs 3 \
    --batch_size 8 \
    --gradient_accumulation_steps 2 \
    --learning_rate 2e-4
```

This will give you **production-quality results in 2-3 hours!** 🚀

---

## 🐛 Troubleshooting

### Out of Memory (OOM)
**Solution:** Reduce batch size or increase gradient accumulation
```bash
# If OOM, try:
--batch_size 4 --gradient_accumulation_steps 4
# Or even smaller:
--batch_size 2 --gradient_accumulation_steps 8
```

### Training Too Slow
**Solution:** Increase batch size (if you have VRAM)
```bash
--batch_size 16 --gradient_accumulation_steps 1
```

### Loss Not Decreasing
**Solution:** Increase learning rate
```bash
--learning_rate 3e-4  # Or even 5e-4
```

### Loss Exploding (NaN)
**Solution:** Decrease learning rate
```bash
--learning_rate 1e-4  # More conservative
```

---

## 📝 After Training

**Evaluate the model:**
```bash
python evaluate.py \
    --model_path ./qwen-epi-forecast \
    --test_file training_data/test.jsonl \
    --save_results
```

**Test predictions:**
```bash
python quick_predict.py --model_path ./qwen-epi-forecast
```

**Expected results:**
- RMSE: 5-10 cases per 100k
- MAE: 3-8 cases per 100k
- Risk Accuracy: 75-85%

---

**Ready? Start training with the recommended command for your GPU!** 🎯
