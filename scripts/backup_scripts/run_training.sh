#!/bin/bash

################################################################################
# Fine-Tuning Training Script for Qwen3-1.7B
# This script runs the training process with configurable parameters
################################################################################

set -e  # Exit on error

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default values
CONFIG_FILE="configs/sft_qwen3_1_7B_improved.yaml"
TRAINING_SCRIPT="src/training/training_script_qwen3_improved.py"
LOG_DIR="logs"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="${LOG_DIR}/training_${TIMESTAMP}.log"

################################################################################
# Helper Functions
################################################################################

print_header() {
    echo -e "${BLUE}================================================================================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}================================================================================================${NC}"
}

print_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

check_gpu() {
    if ! command -v nvidia-smi &> /dev/null; then
        print_error "nvidia-smi not found. CUDA may not be properly installed."
        return 1
    fi
    
    print_info "Checking GPU availability..."
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
    return 0
}

check_dependencies() {
    print_info "Checking Python dependencies..."
    
    # Check if Python is available
    if ! command -v python &> /dev/null; then
        print_error "Python not found. Please install Python 3.8 or higher."
        exit 1
    fi
    
    PYTHON_VERSION=$(python --version | cut -d' ' -f2)
    print_info "Python version: $PYTHON_VERSION"
    
    # Check if required packages are installed
    python -c "import torch; import transformers; import peft; import trl; import datasets" 2>/dev/null
    if [ $? -ne 0 ]; then
        print_warning "Some required packages are missing. Installing dependencies..."
        pip install -r requirements.txt
    else
        print_info "All required packages are installed."
    fi
}

check_wandb() {
    print_info "Checking Weights & Biases setup..."
    
    if [ -z "$WANDB_API_KEY" ]; then
        print_warning "WANDB_API_KEY not set. You may need to login to W&B."
        print_info "Run: wandb login"
        read -p "Do you want to continue without W&B logging? (y/n): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    else
        print_info "W&B API key found."
    fi
}

setup_logging() {
    # Create logs directory if it doesn't exist
    mkdir -p "$LOG_DIR"
    print_info "Logs will be saved to: $LOG_FILE"
}

################################################################################
# Argument Parsing
################################################################################

usage() {
    cat << EOF
Usage: $0 [OPTIONS]

OPTIONS:
    -c, --config FILE           Path to YAML config file (default: $CONFIG_FILE)
    -s, --script FILE           Path to training script (default: $TRAINING_SCRIPT)
    -l, --log-dir DIR           Directory for logs (default: $LOG_DIR)
    --skip-checks               Skip GPU and dependency checks
    --no-wandb                  Disable W&B logging
    -h, --help                  Show this help message

EXAMPLES:
    # Run with default config
    $0

    # Run with custom config
    $0 -c configs/my_config.yaml

    # Run without W&B logging
    $0 --no-wandb

    # Skip all checks (not recommended)
    $0 --skip-checks
EOF
}

SKIP_CHECKS=false
DISABLE_WANDB=false

while [[ $# -gt 0 ]]; do
    case $1 in
        -c|--config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        -s|--script)
            TRAINING_SCRIPT="$2"
            shift 2
            ;;
        -l|--log-dir)
            LOG_DIR="$2"
            LOG_FILE="${LOG_DIR}/training_${TIMESTAMP}.log"
            shift 2
            ;;
        --skip-checks)
            SKIP_CHECKS=true
            shift
            ;;
        --no-wandb)
            DISABLE_WANDB=true
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            usage
            exit 1
            ;;
    esac
done

################################################################################
# Main Execution
################################################################################

print_header "Qwen3-1.7B Fine-Tuning Training"

# Check if files exist
if [ ! -f "$CONFIG_FILE" ]; then
    print_error "Config file not found: $CONFIG_FILE"
    exit 1
fi

if [ ! -f "$TRAINING_SCRIPT" ]; then
    print_error "Training script not found: $TRAINING_SCRIPT"
    exit 1
fi

print_info "Configuration file: $CONFIG_FILE"
print_info "Training script: $TRAINING_SCRIPT"
echo

# Setup logging
setup_logging

# Run checks unless skipped
if [ "$SKIP_CHECKS" = false ]; then
    check_gpu
    check_dependencies
    if [ "$DISABLE_WANDB" = false ]; then
        check_wandb
    fi
    echo
fi

# Disable W&B if requested
if [ "$DISABLE_WANDB" = true ]; then
    export WANDB_MODE=disabled
    print_warning "W&B logging disabled"
fi

# Display training configuration
print_header "Training Configuration"
print_info "Reading configuration from: $CONFIG_FILE"
echo

# Parse and display key configuration values
python - <<EOF
import yaml
try:
    with open("$CONFIG_FILE", 'r') as f:
        config = yaml.safe_load(f)
    
    print(f"Model: {config['model']['name']}")
    print(f"Dataset: {config['dataset']['name']}")
    print(f"LoRA r: {config['lora']['r']}")
    print(f"LoRA alpha: {config['lora']['lora_alpha']}")
    print(f"Batch size: {config['training']['per_device_train_batch_size']}")
    print(f"Gradient accumulation: {config['training']['gradient_accumulation_steps']}")
    print(f"Epochs: {config['training']['num_train_epochs']}")
    print(f"Learning rate: {config['training']['learning_rate']}")
    print(f"Max length: {config['dataset']['max_length']}")
    print(f"Output directory: {config['training']['output_dir']}")
except Exception as e:
    print(f"Error reading config: {e}")
EOF

echo

# Confirm before starting
read -p "$(echo -e ${GREEN}Do you want to start training? [y/N]:${NC} )" -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    print_warning "Training cancelled by user."
    exit 0
fi

print_header "Starting Training"
print_info "Training started at: $(date)"
print_info "Logs are being saved to: $LOG_FILE"
echo

# Run training with logging
python "$TRAINING_SCRIPT" --config "$CONFIG_FILE" 2>&1 | tee "$LOG_FILE"

# Check training result
if [ ${PIPESTATUS[0]} -eq 0 ]; then
    print_header "Training Completed Successfully"
    print_info "Training finished at: $(date)"
    print_info "Logs saved to: $LOG_FILE"
    
    # Display output directory
    OUTPUT_DIR=$(python -c "import yaml; config = yaml.safe_load(open('$CONFIG_FILE')); print(config['paths']['final_model_dir'])")
    print_info "Model saved to: $OUTPUT_DIR"
    
    echo
    print_info "Next steps:"
    echo "  1. Run inference: ./run_inference.sh --model-path $OUTPUT_DIR"
    echo "  2. Run evaluation: ./run_evaluation.sh --model-path $OUTPUT_DIR"
else
    print_header "Training Failed"
    print_error "Training failed. Check logs at: $LOG_FILE"
    exit 1
fi
