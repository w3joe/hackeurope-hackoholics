#!/bin/bash
# Setup script for Linux CLI environment with GPU
# Usage: bash setup_linux.sh

set -e  # Exit on error

echo "========================================="
echo "Qwen Fine-Tuning Setup for Linux CLI"
echo "========================================="

# Check for CUDA
if command -v nvidia-smi &> /dev/null; then
    echo "✓ NVIDIA GPU detected:"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
else
    echo "⚠ Warning: No NVIDIA GPU detected. Training will be very slow on CPU."
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Create virtual environment
echo -e "\n[1/5] Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
echo -e "\n[2/5] Upgrading pip..."
pip install --upgrade pip setuptools wheel

# Install PyTorch with CUDA support
echo -e "\n[3/5] Installing PyTorch with CUDA..."
# Detect CUDA version
if command -v nvcc &> /dev/null; then
    CUDA_VERSION=$(nvcc --version | grep "release" | awk '{print $5}' | cut -d',' -f1)
    echo "Detected CUDA version: $CUDA_VERSION"
fi

# Install PyTorch (adjust for your CUDA version)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install other requirements
echo -e "\n[4/5] Installing dependencies..."
pip install -r requirements.txt

# Verify installation
echo -e "\n[5/5] Verifying installation..."
python3 << EOF
import torch
import transformers
import peft
import trl

print(f"✓ PyTorch version: {torch.__version__}")
print(f"✓ CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"✓ CUDA version: {torch.version.cuda}")
    print(f"✓ GPU: {torch.cuda.get_device_name(0)}")
    print(f"✓ GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
print(f"✓ Transformers version: {transformers.__version__}")
print(f"✓ PEFT version: {peft.__version__}")
print(f"✓ TRL version: {trl.__version__}")
EOF

echo -e "\n========================================="
echo "Setup complete!"
echo "========================================="
echo "To activate the environment:"
echo "  source venv/bin/activate"
echo ""
echo "To start training:"
echo "  python prepare_training_data.py"
echo "  python train_qwen.py --data_dir training_data --output_dir ./qwen-epi-forecast"
