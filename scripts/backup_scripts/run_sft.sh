#!/bin/bash

# SFT Training Script
# Description: Run supervised fine-tuning for Qwen/Llama models using YAML configuration
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

# Change to project root directory first
cd "$PROJECT_ROOT"

# Load .env file if it exists (must be done after cd to project root)
if [ -f ".env" ]; then
    print_info "Loading environment variables from .env file..."
    # Export variables from .env, ignoring comments and empty lines
    export $(grep -v '^#' .env | grep -v '^$' | xargs)
    print_success ".env file loaded"
else
    print_info "No .env file found in project root"
fi

# Default config name
CONFIG_NAME="${1:-sft_qwen3_1_7B.yaml}"

# Define paths
CONFIG_DIR="$PROJECT_ROOT/configs"
TRAINING_DIR="$PROJECT_ROOT/src/training"

# Full path to config file
CONFIG_FILE="$CONFIG_DIR/$CONFIG_NAME"

# Auto-detect the appropriate training script based on config name
if [[ "$CONFIG_NAME" == *"qwen"* ]] || [[ "$CONFIG_NAME" == *"Qwen"* ]]; then
    TRAINING_SCRIPT="$TRAINING_DIR/training_script_qwen3-1.7B_simple.py"
    print_info "Detected Qwen model configuration, using Qwen training script"
elif [[ "$CONFIG_NAME" == *"llama"* ]] || [[ "$CONFIG_NAME" == *"Llama"* ]]; then
    TRAINING_SCRIPT="$TRAINING_DIR/training_script_llama32_simple.py"
    print_info "Detected Llama model configuration, using Llama training script"
else
    # Default to searching for sft_train.py, or use the most recent training script
    if [ -f "$PROJECT_ROOT/src/training/sft_train.py" ]; then
        TRAINING_SCRIPT="$PROJECT_ROOT/src/training/sft_train.py"
    else
        # Try to find any suitable training script
        print_warning "Could not auto-detect model type from config name"
        print_info "Available training scripts in $TRAINING_DIR:"
        ls -1 "$TRAINING_DIR"/*.py 2>/dev/null | grep -E "training_script|sft_train" || echo "  No training scripts found"
        
        # Ask user to specify
        echo ""
        read -p "Please enter the training script filename (e.g., training_script_qwen3-1.7B_simple.py): " script_name
        TRAINING_SCRIPT="$TRAINING_DIR/$script_name"
    fi
fi

# Check if training script exists
if [ ! -f "$TRAINING_SCRIPT" ]; then
    print_error "Training script not found at: $TRAINING_SCRIPT"
    print_info "Available Python files in $TRAINING_DIR:"
    ls -1 "$TRAINING_DIR"/*.py 2>/dev/null || echo "  No Python files found"
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

# Extract model name from config
MODEL_NAME=$(python3 -c "import yaml; config=yaml.safe_load(open('$CONFIG_FILE')); print(config.get('model', {}).get('name', 'Unknown'))" 2>/dev/null || echo "Unknown")
print_info "Model: $MODEL_NAME"

# Check W&B status
if [ ! -z "$WANDB_API_KEY" ]; then
    print_success "W&B API Key is set (first 8 chars: ${WANDB_API_KEY:0:8}...)"
else
    print_warning "WANDB_API_KEY not set. Will run in offline mode."
fi

echo "=========================================="
echo ""

# Check GPU availability
if command -v nvidia-smi &> /dev/null; then
    print_info "GPU Information:"
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
    echo ""
    
    # Check CUDA availability through Python
    print_info "Checking CUDA availability in Python..."
    python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}')" || print_warning "Could not check CUDA availability"
    echo ""
else
    print_warning "nvidia-smi not found. Cannot check GPU status."
fi

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    print_info "Activating virtual environment (venv)..."
    source venv/bin/activate
elif [ -d ".venv" ]; then
    print_info "Activating virtual environment (.venv)..."
    source .venv/bin/activate
elif [ -d "env" ]; then
    print_info "Activating virtual environment (env)..."
    source env/bin/activate
else
    print_warning "No virtual environment found (venv, .venv, or env)"
    print_info "Running with system Python installation"
fi

# Check if required packages are installed
print_info "Checking Python environment..."
python3 --version

# Check for required packages
print_info "Checking required packages..."
MISSING_PACKAGES=()

if ! python3 -c "import transformers" 2>/dev/null; then
    MISSING_PACKAGES+=("transformers")
fi

if ! python3 -c "import torch" 2>/dev/null; then
    MISSING_PACKAGES+=("torch")
fi

if ! python3 -c "import datasets" 2>/dev/null; then
    MISSING_PACKAGES+=("datasets")
fi

if ! python3 -c "import trl" 2>/dev/null; then
    MISSING_PACKAGES+=("trl")
fi

if ! python3 -c "import peft" 2>/dev/null; then
    MISSING_PACKAGES+=("peft")
fi

if ! python3 -c "import yaml" 2>/dev/null; then
    MISSING_PACKAGES+=("pyyaml")
fi

if ! python3 -c "import bitsandbytes" 2>/dev/null; then
    MISSING_PACKAGES+=("bitsandbytes")
fi

if [ ${#MISSING_PACKAGES[@]} -gt 0 ]; then
    print_error "Missing required packages: ${MISSING_PACKAGES[*]}"
    print_info "Please install them using: pip install ${MISSING_PACKAGES[*]}"
    
    # Check if requirements.txt exists
    if [ -f "$PROJECT_ROOT/requirements.txt" ]; then
        print_info "Or install all requirements using: pip install -r requirements.txt"
    fi
    exit 1
else
    print_success "All required packages are installed"
fi

# Create output directories if they don't exist
CONFIG_OUTPUT=$(python3 -c "import yaml; config=yaml.safe_load(open('$CONFIG_FILE')); print(config.get('training', {}).get('output_dir', './output'))" 2>/dev/null || echo "./output")
CONFIG_FINAL=$(python3 -c "import yaml; config=yaml.safe_load(open('$CONFIG_FILE')); print(config.get('paths', {}).get('final_model_dir', './final_model'))" 2>/dev/null || echo "./final_model")

print_info "Creating output directories..."
mkdir -p "$CONFIG_OUTPUT"
mkdir -p "$CONFIG_FINAL"

# Set environment variables for better performance (only if not already set)
if [ -z "$TOKENIZERS_PARALLELISM" ]; then
    export TOKENIZERS_PARALLELISM=false  # Avoid tokenizer warnings
fi

if [ -z "$CUDA_LAUNCH_BLOCKING" ]; then
    export CUDA_LAUNCH_BLOCKING=1  # Better error messages for CUDA errors
fi

# Set W&B mode if API key is not set (only if not already set by .env)
if [ -z "$WANDB_API_KEY" ] && [ -z "$WANDB_MODE" ]; then
    print_warning "Setting WANDB_MODE=offline since no API key found"
    export WANDB_MODE=offline
fi

# Display final environment status
echo ""
echo "=========================================="
echo "  Environment Variables Set"
echo "=========================================="
if [ ! -z "$WANDB_API_KEY" ]; then
    echo "  WANDB_API_KEY: Set ✓"
fi
if [ ! -z "$WANDB_MODE" ]; then
    echo "  WANDB_MODE: $WANDB_MODE"
fi
if [ ! -z "$WANDB_ENTITY" ]; then
    echo "  WANDB_ENTITY: $WANDB_ENTITY"
fi
if [ ! -z "$WANDB_PROJECT" ]; then
    echo "  WANDB_PROJECT: $WANDB_PROJECT"
fi
if [ ! -z "$HF_TOKEN" ]; then
    echo "  HF_TOKEN: Set ✓"
fi
echo "  TOKENIZERS_PARALLELISM: $TOKENIZERS_PARALLELISM"
echo "  CUDA_LAUNCH_BLOCKING: $CUDA_LAUNCH_BLOCKING"
if [ ! -z "$CUDA_VISIBLE_DEVICES" ]; then
    echo "  CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
fi
echo "=========================================="
echo ""

# Run the training script
print_info "Starting training..."
print_info "Command: python3 $TRAINING_SCRIPT --config $CONFIG_FILE"
echo ""

# Create a log file
LOG_FILE="$PROJECT_ROOT/training_$(date +%Y%m%d_%H%M%S).log"
print_info "Logging output to: $LOG_FILE"
echo ""

# Run with proper error handling and logging
if python3 "$TRAINING_SCRIPT" --config "$CONFIG_FILE" 2>&1 | tee "$LOG_FILE"; then
    echo ""
    print_success "Training completed successfully!"
    
    # Print output directory info
    print_info "Model checkpoints saved to: $CONFIG_OUTPUT"
    print_info "Final model saved to: $CONFIG_FINAL"
    
    # Check if the final model was actually saved
    if [ -d "$CONFIG_FINAL" ] && [ "$(ls -A $CONFIG_FINAL)" ]; then
        print_success "Final model directory contains files:"
        ls -la "$CONFIG_FINAL" | head -10
    else
        print_warning "Final model directory is empty or doesn't exist"
    fi
    
    # Print log file location
    print_info "Training log saved to: $LOG_FILE"
else
    echo ""
    print_error "Training failed! Check the error messages above."
    print_info "Full log available at: $LOG_FILE"
    exit 1
fi

echo ""
print_success "Script completed!"
print_info "Next steps:"
print_info "  1. Check training metrics in Weights & Biases (if configured)"
print_info "  2. Evaluate the model using the checkpoint in: $CONFIG_FINAL"
print_info "  3. Review the training log: $LOG_FILE"