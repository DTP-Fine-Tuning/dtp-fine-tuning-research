#!/bin/bash

################################################################################
# Gradio Inference Script for Fine-tuned Models
# This script launches the Gradio web interface for interacting with models
# Supports: Llama, Qwen, Mistral, Gemma, Phi, and other model families
# Must be run from project root: ~/dtp-fine-tuning-research/
################################################################################

set -e  # Exit on error

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default values - Updated for correct directory structure
MODEL_PATH=""  # Will be auto-detected
BASE_MODEL=""
INFERENCE_SCRIPT="src/training/agq_inference.py"
PORT=7860
MAX_NEW_TOKENS=512
SHARE=false
NO_4BIT=false

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
        print_info "Please cd to project root and run: ./scripts/run_inference.sh"
        exit 1
    fi
}

find_latest_model() {
    # Search for models in both src/training and src/utils directories
    local latest_model=""
    local latest_time=0
    
    # Check src/training/SFT-* directories
    if compgen -G "src/training/SFT-*" > /dev/null; then
        for model in src/training/SFT-*/; do
            if [ -d "$model" ]; then
                model_time=$(stat -c %Y "$model" 2>/dev/null || stat -f %m "$model" 2>/dev/null || echo 0)
                if [ "$model_time" -gt "$latest_time" ]; then
                    latest_time=$model_time
                    latest_model=$model
                fi
            fi
        done
    fi
    
    # Check src/utils/SFT-* directories
    if compgen -G "src/utils/SFT-*" > /dev/null; then
        for model in src/utils/SFT-*/; do
            if [ -d "$model" ]; then
                model_time=$(stat -c %Y "$model" 2>/dev/null || stat -f %m "$model" 2>/dev/null || echo 0)
                if [ "$model_time" -gt "$latest_time" ]; then
                    latest_time=$model_time
                    latest_model=$model
                fi
            fi
        done
    fi
    
    echo "$latest_model"
}

list_available_models() {
    print_info "Available models:"
    local count=0
    
    # List models in src/training/
    if compgen -G "src/training/SFT-*" > /dev/null; then
        for model in src/training/SFT-*/; do
            if [ -d "$model" ]; then
                echo "  - $model"
                ((count++))
            fi
        done
    fi
    
    # List models in src/utils/
    if compgen -G "src/utils/SFT-*" > /dev/null; then
        for model in src/utils/SFT-*/; do
            if [ -d "$model" ]; then
                echo "  - $model"
                ((count++))
            fi
        done
    fi
    
    if [ $count -eq 0 ]; then
        echo "  No models found. Please run training first."
        return 1
    fi
    
    return 0
}

check_gpu() {
    if ! command -v nvidia-smi &> /dev/null; then
        print_warning "nvidia-smi not found. Will run on CPU (slow)."
        return 1
    fi
    
    print_info "Checking GPU availability..."
    nvidia-smi --query-gpu=name,memory.free --format=csv,noheader
    return 0
}

check_dependencies() {
    print_info "Checking Python dependencies..."
    
    python -c "import torch; import transformers; import peft; import gradio" 2>/dev/null
    if [ $? -ne 0 ]; then
        print_warning "Some required packages are missing. Installing dependencies..."
        pip install torch transformers peft gradio
    else
        print_info "All required packages are installed."
    fi
}

check_port() {
    if lsof -Pi :$PORT -sTCP:LISTEN -t >/dev/null 2>&1 ; then
        print_warning "Port $PORT is already in use."
        read -p "Do you want to use a different port? [y/N]: " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            read -p "Enter new port number: " PORT
            print_info "Using port: $PORT"
        else
            print_error "Cannot start server on port $PORT."
            exit 1
        fi
    fi
}

get_local_ip() {
    # Try to get local IP address
    LOCAL_IP=$(hostname -I 2>/dev/null | awk '{print $1}')
    if [ -z "$LOCAL_IP" ]; then
        LOCAL_IP="localhost"
    fi
    echo "$LOCAL_IP"
}

################################################################################
# Argument Parsing
################################################################################

usage() {
    cat << EOF
Usage: $0 [OPTIONS]

OPTIONS:
    -m, --model-path PATH       Path to fine-tuned model (auto-detected if not specified)
    -b, --base-model NAME       Base model name (optional, auto-detected from training_info.json)
    -p, --port PORT             Port to run Gradio on (default: $PORT)
    -t, --max-tokens NUM        Max tokens to generate (default: $MAX_NEW_TOKENS)
    -s, --share                 Create public Gradio link
    --no-4bit                   Disable 4-bit quantization
    --script FILE               Path to inference script (default: $INFERENCE_SCRIPT)
    -h, --help                  Show this help message

EXAMPLES:
    # Run with auto-detected latest model
    ./scripts/run_inference.sh

    # Run with specific Qwen model
    ./scripts/run_inference.sh -m src/training/SFT-Qwen3-1.7B-LoRA-final

    # Run with Llama model
    ./scripts/run_inference.sh -m src/training/SFT-Llama3-1B-LoRA-final

    # Run on specific port and share publicly
    ./scripts/run_inference.sh -p 8080 -s

    # Run without 4-bit quantization (requires more VRAM)
    ./scripts/run_inference.sh --no-4bit
EOF
}

while [[ $# -gt 0 ]]; do
    case $1 in
        -m|--model-path)
            MODEL_PATH="$2"
            shift 2
            ;;
        -b|--base-model)
            BASE_MODEL="$2"
            shift 2
            ;;
        -p|--port)
            PORT="$2"
            shift 2
            ;;
        -t|--max-tokens)
            MAX_NEW_TOKENS="$2"
            shift 2
            ;;
        -s|--share)
            SHARE=true
            shift
            ;;
        --no-4bit)
            NO_4BIT=true
            shift
            ;;
        --script)
            INFERENCE_SCRIPT="$2"
            shift 2
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

print_header "Fine-tuned Model Inference - Gradio Interface"

# Verify we're in project root
verify_project_root

# Load environment variables from .env file
load_env_file

# Auto-detect model if not specified
if [ -z "$MODEL_PATH" ]; then
    print_info "No model path specified, searching for models..."
    list_available_models
    
    MODEL_PATH=$(find_latest_model)
    if [ -z "$MODEL_PATH" ]; then
        print_error "No models found. Please train a model first or specify -m option."
        exit 1
    fi
    
    print_info "Auto-detected latest model: $MODEL_PATH"
    read -p "Use this model? [Y/n]: " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Nn]$ ]]; then
        read -p "Enter model path: " MODEL_PATH
    fi
fi

# Remove trailing slash from model path
MODEL_PATH="${MODEL_PATH%/}"

# Check if model path exists
if [ ! -d "$MODEL_PATH" ]; then
    print_error "Model path not found: $MODEL_PATH"
    list_available_models
    exit 1
fi

# Check if inference script exists
if [ ! -f "$INFERENCE_SCRIPT" ]; then
    print_error "Inference script not found: $INFERENCE_SCRIPT"
    print_info "Expected location: src/training/agq_inference.py"
    exit 1
fi

print_info "Model path: $MODEL_PATH"
print_info "Inference script: $INFERENCE_SCRIPT"
print_info "Port: $PORT"
print_info "Max tokens: $MAX_NEW_TOKENS"
print_info "Share publicly: $SHARE"
print_info "4-bit quantization: $([ "$NO_4BIT" = true ] && echo 'Disabled' || echo 'Enabled')"
echo

# Run checks
check_gpu
echo
check_dependencies
echo
check_port
echo

# Build command
CMD="python $INFERENCE_SCRIPT --model-path \"$MODEL_PATH\" --port $PORT --max-new-tokens $MAX_NEW_TOKENS"

if [ -n "$BASE_MODEL" ]; then
    CMD="$CMD --base-model \"$BASE_MODEL\""
fi

if [ "$SHARE" = true ]; then
    CMD="$CMD --share"
fi

if [ "$NO_4BIT" = true ]; then
    CMD="$CMD --no-4bit"
fi

# Display model information
print_header "Model Information"
if [ -f "$MODEL_PATH/training_info.json" ]; then
    python - <<EOF
import json
try:
    with open("$MODEL_PATH/training_info.json", 'r') as f:
        info = json.load(f)

    model_family = info.get('model_family', 'Unknown')
    print(f"Model Family: {model_family.upper() if model_family != 'Unknown' else model_family}")
    print(f"Base Model: {info.get('model_name', 'Unknown')}")
    print(f"Training Completed: {info.get('training_completed', 'Unknown')}")
    print(f"Dataset: {info.get('dataset_info', {}).get('name', 'Unknown')}")
    print(f"Max Length: {info.get('dataset_info', {}).get('max_length', 'Unknown')}")

    # Show LoRA configuration
    lora_config = info.get('lora_config', {})
    if lora_config:
        print(f"LoRA Rank (r): {lora_config.get('r', 'Unknown')}")
        print(f"LoRA Alpha: {lora_config.get('lora_alpha', 'Unknown')}")
except Exception as e:
    print(f"Could not read training info: {e}")
EOF
else
    print_warning "training_info.json not found in model directory"
    # Try to detect from adapter_config.json
    if [ -f "$MODEL_PATH/adapter_config.json" ]; then
        print_info "Found adapter_config.json, reading base model info..."
        python - <<EOF
import json
try:
    with open("$MODEL_PATH/adapter_config.json", 'r') as f:
        config = json.load(f)
    print(f"Base Model: {config.get('base_model_name_or_path', 'Unknown')}")
    print(f"LoRA Rank (r): {config.get('r', 'Unknown')}")
    print(f"LoRA Alpha: {config.get('lora_alpha', 'Unknown')}")
except Exception as e:
    print(f"Could not read adapter config: {e}")
EOF
    fi
fi
echo

# Start server
print_header "Starting Gradio Server"
print_info "Launching Gradio interface..."
print_info "Please wait, loading model..."
echo

# Display access URLs
LOCAL_IP=$(get_local_ip)
print_info "Server will be accessible at:"
echo "  - Local: http://localhost:$PORT"
echo "  - Network: http://$LOCAL_IP:$PORT"
if [ "$SHARE" = true ]; then
    echo "  - Public: Will be shown after launch"
fi
echo

print_warning "Press Ctrl+C to stop the server"
echo

# Run the inference script
eval $CMD

# This will only run if the server is stopped
print_header "Server Stopped"
print_info "Gradio interface closed."