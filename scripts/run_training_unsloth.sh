#!/bin/bash
#unsloth training script
#author: Tim 2 DTP


set -e 
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' 

#def values
CONFIG_FILE="configs/test/sft_qwen3_unsloth_test.yaml"
TRAINING_SCRIPT="src/training/train_unsloth_multi-turn.py"

#if u wanna train for single turn use train_unsloth_single_turn.py for single-turn datasets
# TRAINING_SCRIPT="src/training/train_unsloth_single_turn.py"
LOG_DIR="logs"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="${LOG_DIR}/training_unsloth_${TIMESTAMP}.log"

print_header() {
    echo -e "${CYAN}================================================================================================${NC}"
    echo -e "${CYAN}$1${NC}"
    echo -e "${CYAN}================================================================================================${NC}"
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

load_env_file() {
    if [ -f ".env" ]; then
        print_info "Loading environment variables from .env file..."
        set -a
        source .env
        set +a
    fi
}

verify_project_root() {
    if [ ! -d "scripts" ] || [ ! -d "src" ] || [ ! -d "configs" ]; then
        print_error "This script must be run from the project root directory"
        print_info "Current directory: $(pwd)"
        print_info "Please cd to project root and run: ./scripts/run_training_unsloth.sh"
        exit 1
    fi
}

check_gpu() {
    if ! command -v nvidia-smi &> /dev/null; then
        print_error "nvidia-smi not found. CUDA may not be properly installed."
        return 1
    fi
    
    print_info "Checking GPU availability..."
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader    
    CUDA_VERSION=$(nvidia-smi | grep "CUDA Version" | awk '{print $9}')
    print_info "CUDA Version: $CUDA_VERSION"
    return 0
}

check_unsloth_dependencies() {
    print_info "Checking Unsloth dependencies..."
    
    if ! command -v python &> /dev/null; then
        print_error "Python not found. Please install Python 3.8 or higher."
        exit 1
    fi
    
    PYTHON_VERSION=$(python --version | cut -d' ' -f2)
    print_info "Python version: $PYTHON_VERSION"    
    print_info "Verifying Unsloth installation..."
    python -c "from unsloth import FastLanguageModel; print('  [DONE] Unsloth installed')" 2>/dev/null || {
        print_error "Unsloth not installed. Please install with:"
        echo "  pip install unsloth"
        exit 1
    }
    
    python -c "import torch; print(f'  [DONE] PyTorch {torch.__version__}')" 2>/dev/null || {
        print_error "PyTorch not installed"
        exit 1
    }
    
    python -c "import transformers; print(f'  [DONE] Transformers {transformers.__version__}')" 2>/dev/null || {
        print_error "Transformers not installed"
        exit 1
    }
    
    python -c "import trl; print(f'  [DONE] TRL {trl.__version__}')" 2>/dev/null || {
        print_error "TRL not installed"
        exit 1
    }
    
    python -c "import peft; print(f'  [DONE] PEFT {peft.__version__}')" 2>/dev/null || {
        print_error "PEFT not installed"
        exit 1
    }
    
    python -c "import datasets; print(f'  [DONE] Datasets {datasets.__version__}')" 2>/dev/null || {
        print_error "Datasets not installed"
        exit 1
    }
    
    print_info "All Unsloth dependencies verified."
}

check_wandb() {
    print_info "Checking Weights & Biases setup..."
    
    if [ -z "$WANDB_API_KEY" ]; then
        print_warning "WANDB_API_KEY not set."
        print_info "You can add it to .env file or run: wandb login"
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
    mkdir -p "$LOG_DIR"
    print_info "Logs will be saved to: $LOG_FILE"
}

usage() {
    cat << EOF
Usage: $0 [OPTIONS]

Unsloth Training Script - Optimized for Qwen3 models

OPTIONS:
    -c, --config FILE           Path to YAML config file (default: $CONFIG_FILE)
    -s, --script FILE           Path to training script (default: $TRAINING_SCRIPT)
    -l, --log-dir DIR           Directory for logs (default: $LOG_DIR)
    --skip-checks               Skip GPU and dependency checks
    --no-wandb                  Disable W&B logging
    -h, --help                  Show this help message

EXAMPLES:
    # Run with default Qwen3 Unsloth config
    ./scripts/run_training_unsloth.sh

    # Run with custom config
    ./scripts/run_training_unsloth.sh -c configs/sft_diploy_8B.yaml

    # Run without W&B logging
    ./scripts/run_training_unsloth.sh --no-wandb

    # Skip all checks (not recommended)
    ./scripts/run_training_unsloth.sh --skip-checks

NOTE:
    For generic training without Unsloth, use: ./scripts/run_training.sh
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
            LOG_FILE="${LOG_DIR}/training_unsloth_${TIMESTAMP}.log"
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


print_header "Unsloth Fine-Tuning Training (Qwen3 Optimized)"
verify_project_root
load_env_file

if [ ! -f "$CONFIG_FILE" ]; then
    print_error "Config file not found: $CONFIG_FILE"
    print_info "Available Unsloth configs:"
    ls -1 configs/*unsloth*.yaml 2>/dev/null || echo "  No Unsloth config files found"
    print_info "All available configs:"
    ls -1 configs/*.yaml 2>/dev/null || echo "  No config files found"
    exit 1
fi

if [ ! -f "$TRAINING_SCRIPT" ]; then
    print_error "Training script not found: $TRAINING_SCRIPT"
    print_info "Expected location: src/training/train_unsloth_multi-turn.py"
    exit 1
fi

print_info "Configuration file: $CONFIG_FILE"
print_info "Training script: $TRAINING_SCRIPT"
echo
setup_logging

if [ "$SKIP_CHECKS" = false ]; then
    check_gpu
    echo
    check_unsloth_dependencies
    echo
    if [ "$DISABLE_WANDB" = false ]; then
        check_wandb
    fi
    echo
fi

if [ "$DISABLE_WANDB" = true ]; then
    export WANDB_MODE=disabled
    print_warning "W&B logging disabled"
fi

print_header "Training Configuration (Unsloth)"
print_info "Reading configuration from: $CONFIG_FILE"
echo
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
    effective_batch = config['training']['per_device_train_batch_size'] * config['training']['gradient_accumulation_steps']
    print(f"Effective batch size: {effective_batch}")
    print(f"Epochs: {config['training']['num_train_epochs']}")
    print(f"Learning rate: {config['training']['learning_rate']}")
    print(f"Max length: {config['dataset']['max_length']}")
    print(f"Output directory: {config['training']['output_dir']}")
    
    # Unsloth specific info
    if config.get('quantization', {}).get('load_in_4bit', False):
        print(f"4-bit quantization: Enabled (memory efficient)")
    
    # Advanced settings
    advanced = config.get('advanced', {})
    print(f"\n[Advanced Settings]")
    print(f"  Max grad norm: {advanced.get('max_grad_norm', 1.0)}")
    print(f"  NEFTune: {'Enabled (alpha={})'.format(advanced.get('neftune_noise_alpha', 5.0)) if advanced.get('use_neftune', False) else 'Disabled'}")
    print(f"  Packing: {'Enabled' if config['training'].get('packing', False) else 'Disabled'}")
    print(f"  Flash Attention: {'Enabled' if advanced.get('use_flash_attention', True) else 'Disabled'} (via Unsloth)")
    
    print(f"\nFinal model dir: {config.get('paths', {}).get('final_model_dir', 'N/A')}")
    
except Exception as e:
    print(f"Error reading config: {e}")
EOF

echo
read -p "$(echo -e ${GREEN}Do you want to start Unsloth training? [y/N]:${NC} )" -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    print_warning "Training cancelled by user."
    exit 0
fi

print_header "Starting Unsloth Training"
print_info "Training started at: $(date)"
print_info "Logs are being saved to: $LOG_FILE"
print_info "Using Unsloth for optimized training speed..."
echo
python "$TRAINING_SCRIPT" --config "$CONFIG_FILE" 2>&1 | tee "$LOG_FILE"

if [ ${PIPESTATUS[0]} -eq 0 ]; then
    print_header "Unsloth Training Completed Successfully"
    print_info "Training finished at: $(date)"
    print_info "Logs saved to: $LOG_FILE"
    OUTPUT_DIR=$(python -c "import yaml; config = yaml.safe_load(open('$CONFIG_FILE')); print(config.get('paths', {}).get('final_model_dir', config['training']['output_dir']))" 2>/dev/null)
    
    if [ -n "$OUTPUT_DIR" ] && [ -d "$OUTPUT_DIR" ]; then
        print_info "Model saved to: $OUTPUT_DIR"
        echo
        print_info "Model files:"
        ls -lh "$OUTPUT_DIR" 2>/dev/null | head -10
        echo
        print_info "Next steps:"
        echo "  1. Run inference: ./scripts/run_inference.sh -m $OUTPUT_DIR"
        echo "  2. Run evaluation (OpenAI): ./scripts/run_evaluation.sh -m $OUTPUT_DIR"
        echo "  3. Run evaluation (Gemini): ./scripts/run_evaluation_gemini.sh -m $OUTPUT_DIR"
        echo "  4. Check W&B dashboard for training metrics and model artifact"
    fi
else
    print_header "Unsloth Training Failed"
    print_error "Training failed. Check logs at: $LOG_FILE"
    print_info "Common issues:"
    echo "  - OOM: Reduce batch_size or max_length in config"
    echo "  - Missing chat_template: Add chat_template section to config"
    echo "  - Dataset format: Ensure dataset has 'text' column"
    exit 1
fi
