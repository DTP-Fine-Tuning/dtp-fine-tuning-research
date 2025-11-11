#!/bin/bash

# Script to run Qwen2 1.5B SFT training
# Usage: bash scripts/run_qwen2_training.sh

set -e

echo "=================================================="
echo "Qwen2 1.5B SFT Training Pipeline"
echo "=================================================="

# Activate virtual environment if needed
# source venv/bin/activate

# Set Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Configuration file
CONFIG_FILE="configs/sft_qwen2_1_5B_1k.yaml"

# Check if config exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Configuration file not found: $CONFIG_FILE"
    exit 1
fi

echo "Using configuration: $CONFIG_FILE"
echo ""

# Run training
python src/training/training_script_qwen2_sft.py \
    --config "$CONFIG_FILE"

echo ""
echo "=================================================="
echo "Training completed successfully!"
echo "=================================================="
