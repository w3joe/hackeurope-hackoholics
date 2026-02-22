#!/bin/bash
# Storage cleanup script for university servers with quota limits
# Run: bash cleanup_storage.sh

echo "========================================="
echo "Storage Cleanup & Analysis"
echo "========================================="

# Check current usage
echo -e "\n[1] Current Disk Usage:"
df -h /cs/student/ug

# Find what's using space
echo -e "\n[2] Top 10 largest directories in your home:"
du -sh ~/* 2>/dev/null | sort -hr | head -10

echo -e "\n[3] Checking common culprits..."

# Check Python caches
if [ -d ~/.cache/pip ]; then
    echo "  - pip cache: $(du -sh ~/.cache/pip 2>/dev/null | cut -f1)"
fi

if [ -d ~/.cache/huggingface ]; then
    echo "  - Hugging Face cache: $(du -sh ~/.cache/huggingface 2>/dev/null | cut -f1)"
fi

if [ -d ~/.cache/torch ]; then
    echo "  - PyTorch cache: $(du -sh ~/.cache/torch 2>/dev/null | cut -f1)"
fi

# Check virtual environments
echo -e "\n[4] Virtual environments found:"
find ~ -maxdepth 3 -name "venv" -o -name ".venv" -o -name "env" 2>/dev/null | while read dir; do
    echo "  - $dir: $(du -sh "$dir" 2>/dev/null | cut -f1)"
done

# Check node modules
echo -e "\n[5] Node modules found:"
find ~ -maxdepth 3 -name "node_modules" 2>/dev/null | while read dir; do
    echo "  - $dir: $(du -sh "$dir" 2>/dev/null | cut -f1)"
done

echo -e "\n========================================="
echo "Cleanup Options:"
echo "========================================="
echo "1. Clear pip cache:          pip cache purge"
echo "2. Clear HF cache:           rm -rf ~/.cache/huggingface/hub/*"
echo "3. Clear torch cache:        rm -rf ~/.cache/torch/*"
echo "4. Remove old venv:          rm -rf ~/path/to/venv"
echo "5. Clear all caches:         rm -rf ~/.cache/*"
echo ""
echo "Run cleanup? (y/n)"
read -r response

if [[ "$response" =~ ^[Yy]$ ]]; then
    echo -e "\n🧹 Starting cleanup..."

    # Clear pip cache
    if command -v pip &> /dev/null; then
        echo "  Clearing pip cache..."
        pip cache purge 2>/dev/null || true
    fi

    # Clear HF cache
    if [ -d ~/.cache/huggingface/hub ]; then
        echo "  Clearing Hugging Face cache..."
        rm -rf ~/.cache/huggingface/hub/*
    fi

    # Clear torch cache
    if [ -d ~/.cache/torch ]; then
        echo "  Clearing PyTorch cache..."
        rm -rf ~/.cache/torch/*
    fi

    echo -e "\n✓ Cleanup complete!"
    echo -e "\nNew disk usage:"
    df -h /cs/student/ug
else
    echo "Cleanup cancelled."
fi

echo -e "\n========================================="
