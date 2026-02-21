# Qwen2.5-3B-Instruct Fine-Tuning for Epidemiological Forecasting

This directory contains a complete pipeline for fine-tuning Qwen2.5-3B-Instruct to predict disease risk and provide 12-week forecasts based on epidemiological surveillance data.

## 📁 Files

- `llm_risk_training.csv` - Training dataset with historical disease data
- `llm_risk_training.jsonl` - JSONL version of training data
- `llm_risk_csv.py` - Script that generated the training data
- `prepare_training_data.py` - Converts CSV to instruction-tuning format
- `train_qwen.py` - QLoRA fine-tuning script
- `inference.py` - Inference script for predictions
- `evaluate.py` - Evaluation script with metrics (RMSE, MAE, accuracy)
- `requirements.txt` - Python dependencies

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

**Hardware Requirements:**
- GPU with 12GB+ VRAM (RTX 3090, 4090, or better)
- CUDA 11.8+ installed
- ~20GB disk space for model and checkpoints

### 2. Prepare Training Data

Convert the CSV dataset into instruction-tuning format:

```bash
python prepare_training_data.py
```

This creates:
- `training_data/train.jsonl` (~80% of data)
- `training_data/val.jsonl` (~5% of data)
- `training_data/test.jsonl` (~15% of data)
- `training_data/sample.json` (inspect this to see the format)

### 3. Fine-Tune the Model

Start training with QLoRA (4-bit quantization):

```bash
python train_qwen.py \
    --data_dir training_data \
    --output_dir ./qwen-epi-forecast \
    --num_epochs 3 \
    --batch_size 4 \
    --gradient_accumulation_steps 4 \
    --learning_rate 2e-4
```

**Training time:** ~2-4 hours on RTX 4090 (depends on dataset size)

**Memory usage:** ~12-16GB VRAM

### 4. Run Inference

#### Interactive Mode
```bash
python inference.py --model_path ./qwen-epi-forecast --interactive
```

#### Single Example
```bash
python inference.py --model_path ./qwen-epi-forecast
```

#### Batch Inference
```bash
python inference.py \
    --model_path ./qwen-epi-forecast \
    --test_file training_data/test.jsonl \
    --output_file results.json
```

### 5. Evaluate the Model

Run evaluation on the test set:

```bash
python evaluate.py \
    --model_path ./qwen-epi-forecast \
    --test_file training_data/test.jsonl \
    --save_results
```

This computes:
- **RMSE & MAE** for 12-week forecast accuracy
- **Classification accuracy** for risk level (LOW/MEDIUM/HIGH/CRITICAL)
- **RMSE & MAE** for spread likelihood prediction

## 📊 Output Format

The model generates structured JSON predictions:

```json
{
  "RiskAssessment": {
    "country": "Austria",
    "risk_level": "HIGH",
    "spread_likelihood": 0.725,
    "reasoning": "Based on elevated TDA status with H0 rising and H1 rising trends, strong signal convergence detected. Forecasting 42.3 peak cases per 100k over the next 12 weeks.",
    "recommended_disease_focus": ["Influenza"],
    "twelve_week_forecast": {
      "weekly_cases_per_100k": [18.8, 25.6, 30.8, 33.9, 34.0, 24.1, 26.2, 13.5, 13.3, 17.4, 3.9, 6.5],
      "forecast_start_week": "2024-W02"
    }
  }
}
```

## 🎯 Model Architecture

- **Base Model:** Qwen2.5-3B-Instruct (3 billion parameters)
- **Fine-Tuning Method:** QLoRA (4-bit quantization)
- **LoRA Configuration:**
  - Rank (r): 16
  - Alpha: 32
  - Dropout: 0.05
  - Target modules: All attention and MLP layers
- **Trainable Parameters:** ~5.6M (0.19% of total)

## 📈 Training Configuration

| Hyperparameter | Default Value | Description |
|---------------|---------------|-------------|
| Epochs | 3 | Number of training passes |
| Batch Size | 4 | Per-device batch size |
| Gradient Accumulation | 4 | Effective batch = 16 |
| Learning Rate | 2e-4 | Peak learning rate |
| LR Scheduler | Cosine | With 10% warmup |
| Optimizer | Paged AdamW 8-bit | Memory-efficient optimizer |
| Max Sequence Length | 2048 tokens | Context window |

## 📝 Input Features

The model is trained on the following features:

1. **Recent Observations:** Last 6 weeks of disease activity
2. **Holt-Winters Forecast:** Statistical forecast for next 12 weeks
3. **TDA (Topological Data Analysis):**
   - Anomaly status (normal/elevated/anomaly)
   - Trend direction (rising/falling)
   - Z-scores for H0 and H1 persistence
4. **Seasonality:** Historical peak weeks
5. **Pattern Analysis:**
   - PCA distance and position
   - Residual z-scores
   - Signal convergence indicator

## 🔧 Advanced Usage

### Custom Training

Fine-tune with custom hyperparameters:

```bash
python train_qwen.py \
    --model_name Qwen/Qwen2.5-3B-Instruct \
    --data_dir training_data \
    --output_dir ./custom-model \
    --num_epochs 5 \
    --batch_size 2 \
    --gradient_accumulation_steps 8 \
    --learning_rate 1e-4
```

### Temperature Adjustment

For more conservative predictions (evaluation):
```python
# In inference.py, set temperature=0.1
response = generate_forecast(model, tokenizer, prompt, temperature=0.1)
```

For more diverse predictions (exploration):
```python
# In inference.py, set temperature=0.7
response = generate_forecast(model, tokenizer, prompt, temperature=0.7)
```

## 📊 Expected Performance

Based on the training data characteristics:

- **Forecast RMSE:** 5-15 cases per 100k (varies by country/pathogen)
- **Forecast MAE:** 3-10 cases per 100k
- **Risk Classification Accuracy:** 70-85%
- **Spread Likelihood MAE:** 0.05-0.15

Note: Performance depends on:
- Training data quality and quantity
- Pathogen characteristics (seasonal vs. endemic)
- Country-specific patterns

## 🐛 Troubleshooting

### Out of Memory (OOM)

Reduce batch size or increase gradient accumulation:
```bash
python train_qwen.py --batch_size 2 --gradient_accumulation_steps 8
```

### Slow Training

Enable gradient checkpointing (already enabled by default) or use mixed precision:
- bf16 (recommended for Ampere+ GPUs)
- fp16 (fallback for older GPUs)

### Model Not Loading

Ensure you have the base model downloaded:
```python
from transformers import AutoModelForCausalLM
AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-3B-Instruct")
```

## 📚 Dataset Information

- **Source:** ECDC (European Centre for Disease Prevention and Control)
- **Pathogens:** Influenza, RSV, SARS-CoV-2
- **Countries:** 30+ European countries
- **Time Range:** 2023-2026 (weekly data)
- **Training Examples:** ~3000+ historical snapshots (with historical expansion)

## 🔗 References

- [Qwen2.5 Model Card](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct)
- [QLoRA Paper](https://arxiv.org/abs/2305.14314)
- [PEFT Documentation](https://huggingface.co/docs/peft)
- [TRL (Transformer Reinforcement Learning)](https://github.com/huggingface/trl)

## 📄 License

This fine-tuning code is provided as-is. Please ensure compliance with:
- Qwen model license
- ECDC data usage terms
- Your institutional data policies

## 🤝 Contributing

Improvements welcome! Consider:
- Hyperparameter optimization
- Multi-pathogen joint training
- Attention visualization
- Confidence interval estimation

---

**Questions?** Check the sample files or run scripts with `--help` for more options.
