#!/bin/bash
# Complete training pipeline script for Linux CLI
# Usage: bash run_training.sh

set -e  # Exit on error

echo "========================================="
echo "Qwen2.5-3B Fine-Tuning Pipeline"
echo "========================================="

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    source venv/bin/activate
    echo "✓ Virtual environment activated"
else
    echo "⚠ Virtual environment not found. Run setup_linux.sh first."
    exit 1
fi

# Step 1: Prepare training data
echo -e "\n[Step 1/3] Preparing training data..."
if [ ! -f "llm_risk_training.csv" ]; then
    echo "Error: llm_risk_training.csv not found!"
    exit 1
fi

python prepare_training_data.py

# Check if data was created
if [ ! -d "training_data" ]; then
    echo "Error: Training data directory not created!"
    exit 1
fi

echo "✓ Training data prepared"
echo "  - Train set: $(wc -l < training_data/train.jsonl) examples"
echo "  - Val set:   $(wc -l < training_data/val.jsonl) examples"
echo "  - Test set:  $(wc -l < training_data/test.jsonl) examples"

# Step 2: Train the model
echo -e "\n[Step 2/3] Starting fine-tuning..."
echo "This will take 2-4 hours on RTX 4090 (varies by GPU and dataset size)"
echo "Press Ctrl+C within 5 seconds to cancel..."
sleep 5

python train_qwen.py \
    --data_dir training_data \
    --output_dir ./qwen-epi-forecast \
    --num_epochs 3 \
    --batch_size 4 \
    --gradient_accumulation_steps 4 \
    --learning_rate 2e-4

# Step 3: Evaluate the model
echo -e "\n[Step 3/3] Evaluating model on test set..."
python evaluate.py \
    --model_path ./qwen-epi-forecast \
    --test_file training_data/test.jsonl \
    --save_results

echo -e "\n========================================="
echo "Training pipeline complete!"
echo "========================================="
echo "Model saved to: ./qwen-epi-forecast"
echo "Evaluation results: ./qwen-epi-forecast/evaluation_results.json"
echo ""
echo "To run inference:"
echo "  python inference.py --model_path ./qwen-epi-forecast --interactive"
