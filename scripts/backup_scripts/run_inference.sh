#!/bin/bash

################################################################################
# Gradio Inference Script for Qwen3 Fine-tuned Model
# This script launches the Gradio web interface for interacting with the model
################################################################################

set -e  # Exit on error

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default values
MODEL_PATH="./SFT-Qwen3-1.7B-LoRA-MultiTurn-final"
BASE_MODEL=""
INFERENCE_SCRIPT="gradio_inference.py"
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
    -m, --model-path PATH       Path to fine-tuned model (default: $MODEL_PATH)
    -b, --base-model NAME       Base model name (optional, auto-detected from training_info.json)
    -p, --port PORT             Port to run Gradio on (default: $PORT)
    -t, --max-tokens NUM        Max tokens to generate (default: $MAX_NEW_TOKENS)
    -s, --share                 Create public Gradio link
    --no-4bit                   Disable 4-bit quantization
    --script FILE               Path to inference script (default: $INFERENCE_SCRIPT)
    -h, --help                  Show this help message

EXAMPLES:
    # Run with default settings
    $0

    # Run with custom model path
    $0 -m ./my-model

    # Run on specific port and share publicly
    $0 -p 8080 -s

    # Run without 4-bit quantization (requires more VRAM)
    $0 --no-4bit

    # Specify base model explicitly
    $0 -m ./my-model -b Qwen/Qwen3-1.7B
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

print_header "Qwen3 Model Inference - Gradio Interface"

# Check if model path exists
if [ ! -d "$MODEL_PATH" ]; then
    print_error "Model path not found: $MODEL_PATH"
    print_info "Available models in current directory:"
    ls -d SFT-* 2>/dev/null || echo "  No models found"
    exit 1
fi

# Check if inference script exists
if [ ! -f "$INFERENCE_SCRIPT" ]; then
    print_error "Inference script not found: $INFERENCE_SCRIPT"
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
    
    print(f"Base Model: {info.get('model_name', 'Unknown')}")
    print(f"Training Completed: {info.get('training_completed', 'Unknown')}")
    print(f"Dataset: {info.get('dataset_info', {}).get('name', 'Unknown')}")
    print(f"Max Length: {info.get('dataset_info', {}).get('max_length', 'Unknown')}")
except Exception as e:
    print(f"Could not read training info: {e}")
EOF
else
    print_warning "training_info.json not found in model directory"
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
