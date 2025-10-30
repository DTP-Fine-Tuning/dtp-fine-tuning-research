#!/bin/bash

# SFT Training Script
# Description: Run supervised fine-tuning for Llama models using YAML configuration
# Usage: bash scripts/run_sft.sh [config_name]

set -e  # Exit on error

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored messages
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Get project root (parent of scripts directory)
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"

print_info "Script directory: $SCRIPT_DIR"
print_info "Project root: $PROJECT_ROOT"

# Define paths
TRAINING_SCRIPT="$PROJECT_ROOT/src/training/sft_train.py"
CONFIG_DIR="$PROJECT_ROOT/configs"

# Default config name
CONFIG_NAME="${1:-sft_llama32_1b.yaml}"

# Full path to config file
CONFIG_FILE="$CONFIG_DIR/$CONFIG_NAME"

# Check if training script exists
if [ ! -f "$TRAINING_SCRIPT" ]; then
    print_error "Training script not found at: $TRAINING_SCRIPT"
    exit 1
fi

# Check if config file exists
if [ ! -f "$CONFIG_FILE" ]; then
    print_error "Configuration file not found at: $CONFIG_FILE"
    print_info "Available configs in $CONFIG_DIR:"
    ls -1 "$CONFIG_DIR"/*.yaml 2>/dev/null || echo "  No config files found"
    exit 1
fi

# Print configuration
echo ""
echo "=========================================="
echo "  SFT Training Configuration"
echo "=========================================="
print_info "Training script: $TRAINING_SCRIPT"
print_info "Config file: $CONFIG_FILE"
print_info "Working directory: $PROJECT_ROOT"
echo "=========================================="
echo ""

# Check GPU availability
if command -v nvidia-smi &> /dev/null; then
    print_info "GPU Information:"
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
    echo ""
else
    print_warning "nvidia-smi not found. Cannot check GPU status."
fi

# Change to project root directory
cd "$PROJECT_ROOT"

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    print_info "Activating virtual environment..."
    source venv/bin/activate
elif [ -d ".venv" ]; then
    print_info "Activating virtual environment..."
    source .venv/bin/activate
else
    print_warning "No virtual environment found (venv or .venv)"
fi

# Check if required packages are installed
print_info "Checking Python environment..."
python3 --version

# Check if required packages are available
if ! python3 -c "import transformers" 2>/dev/null; then
    print_error "transformers package not found. Please install requirements.txt"
    exit 1
fi

# Run the training script
print_info "Starting training..."
print_info "Command: python3 $TRAINING_SCRIPT --config $CONFIG_FILE"
echo ""

# Run with proper error handling
if python3 "$TRAINING_SCRIPT" --config "$CONFIG_FILE"; then
    echo ""
    print_success "Training completed successfully!"
    
    # Print output directory info
    CONFIG_OUTPUT=$(python3 -c "import yaml; config=yaml.safe_load(open('$CONFIG_FILE')); print(config['training']['output_dir'])" 2>/dev/null || echo "./output")
    print_info "Model checkpoints saved to: $CONFIG_OUTPUT"
    
    CONFIG_FINAL=$(python3 -c "import yaml; config=yaml.safe_load(open('$CONFIG_FILE')); print(config['paths']['final_model_dir'])" 2>/dev/null || echo "./final_model")
    print_info "Final model saved to: $CONFIG_FINAL"
else
    echo ""
    print_error "Training failed! Check the error messages above."
    exit 1
fi

echo ""
print_success "Script completed!"
