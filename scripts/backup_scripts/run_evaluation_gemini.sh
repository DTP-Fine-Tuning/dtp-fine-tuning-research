#!/bin/bash

################################################################################
# DeepEval Evaluation Script for Qwen3 Fine-tuned Model with Google Gemini
# This script evaluates the model using various metrics powered by Gemini AI
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
EVALUATION_SCRIPT="deepeval_evaluation_gemini.py"
OUTPUT_DIR="./evaluation_results"
DATASET_NAME=""
DATASET_PATH=""
USE_SAMPLE_DATA=false
TEST_SIZE=100
NO_4BIT=false
METRICS=""
GEMINI_MODEL="gemini-1.5-flash"

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
        print_warning "nvidia-smi not found. Evaluation will run on CPU (very slow)."
        return 1
    fi
    
    print_info "Checking GPU availability..."
    nvidia-smi --query-gpu=name,memory.free --format=csv,noheader
    return 0
}

check_dependencies() {
    print_info "Checking Python dependencies..."
    
    # Check core dependencies
    python -c "import torch; import transformers; import peft; import datasets" 2>/dev/null
    if [ $? -ne 0 ]; then
        print_warning "Some core packages are missing. Installing dependencies..."
        pip install torch transformers peft datasets pandas
    else
        print_info "Core packages are installed."
    fi
    
    # Check DeepEval
    python -c "import deepeval" 2>/dev/null
    if [ $? -ne 0 ]; then
        print_warning "DeepEval not found. Installing..."
        pip install deepeval
    fi
    
    # Check Google Generative AI
    python -c "import google.generativeai" 2>/dev/null
    if [ $? -ne 0 ]; then
        print_warning "Google Generative AI library not found. Installing..."
        pip install google-generativeai
    fi
    
    # Check additional dependencies
    python -c "import tenacity" 2>/dev/null
    if [ $? -ne 0 ]; then
        print_warning "Tenacity not found. Installing..."
        pip install tenacity
    fi
    
    print_info "All required packages are installed."
}

check_gemini_key() {
    print_info "Checking Google Gemini API configuration..."
    
    # Check for Gemini API key in environment
    if [ -z "$GEMINI_API_KEY" ] && [ -z "$GOOGLE_API_KEY" ]; then
        print_warning "Neither GEMINI_API_KEY nor GOOGLE_API_KEY environment variable is set."
        print_warning "DeepEval uses Google Gemini for evaluation metrics."
        echo
        print_info "To get a Gemini API key:"
        echo "  1. Visit: https://makersuite.google.com/app/apikey"
        echo "  2. Sign in with your Google account"
        echo "  3. Create a new API key"
        echo "  4. Set it as an environment variable:"
        echo "     export GEMINI_API_KEY='your-key-here'"
        echo "     # or"
        echo "     export GOOGLE_API_KEY='your-key-here'"
        echo
        
        read -p "Do you want to continue without Gemini API key? (Evaluation will fail) [y/N]: " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    else
        if [ ! -z "$GEMINI_API_KEY" ]; then
            print_info "GEMINI_API_KEY found in environment."
        else
            print_info "GOOGLE_API_KEY found in environment."
        fi
        
        # Validate the API key
        print_info "Validating Gemini API key..."
        python -c "
import os
import sys
try:
    import google.generativeai as genai
    api_key = os.environ.get('GEMINI_API_KEY') or os.environ.get('GOOGLE_API_KEY')
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-1.5-flash')
    response = model.generate_content('Hello, test')
    if response.text:
        print('✓ Gemini API key validated successfully!')
        sys.exit(0)
    else:
        print('✗ Gemini API key validation failed: No response')
        sys.exit(1)
except Exception as e:
    print(f'✗ Gemini API key validation failed: {e}')
    sys.exit(1)
" 2>/dev/null
        
        if [ $? -eq 0 ]; then
            print_info "Gemini API key is valid and working!"
        else:
            print_error "Gemini API key validation failed."
            print_info "Please check your API key and try again."
            read -p "Do you want to continue anyway? [y/N]: " -n 1 -r
            echo
            if [[ ! $REPLY =~ ^[Yy]$ ]]; then
                exit 1
            fi
        fi
    fi
}

################################################################################
# Argument Parsing
################################################################################

usage() {
    cat << EOF
Usage: $0 [OPTIONS]

OPTIONS:
    -m, --model-path PATH       Path to fine-tuned model (default: $MODEL_PATH)
    -b, --base-model NAME       Base model name (optional)
    -o, --output-dir DIR        Output directory for results (default: $OUTPUT_DIR)
    -d, --dataset NAME          HuggingFace dataset name for evaluation
    -f, --dataset-path PATH     Local dataset path for evaluation
    -s, --use-sample-data       Use built-in sample test data (Indonesian examples)
    -n, --test-size NUM         Number of samples to evaluate (default: $TEST_SIZE)
    --gemini-model MODEL        Gemini model to use (default: $GEMINI_MODEL)
                                Options: gemini-1.5-pro, gemini-1.5-flash, gemini-1.0-pro
    --metrics METRICS           Specific metrics to use (space-separated)
                                Available: correctness coherence relevancy 
                                          context_consistency hallucination
    --no-4bit                   Disable 4-bit quantization
    --script FILE               Path to evaluation script (default: $EVALUATION_SCRIPT)
    -h, --help                  Show this help message

GEMINI CONFIGURATION:
    The evaluation uses Google Gemini as the LLM judge for metrics.
    Set one of these environment variables before running:
        export GEMINI_API_KEY='your-api-key'
        export GOOGLE_API_KEY='your-api-key'
    
    Get your API key from: https://makersuite.google.com/app/apikey

GEMINI MODEL OPTIONS:
    - gemini-1.5-flash: Fast, cost-effective model (recommended for most evaluations)
    - gemini-1.5-pro: More capable but slower and costlier
    - gemini-1.0-pro: Legacy model, stable but less capable

EXAMPLES:
    # Evaluate with sample data (quick test with Indonesian examples)
    $0 -m ./my-model -s

    # Evaluate on specific dataset with Gemini Pro
    $0 -m ./my-model -d "izzulgod/indonesian-conversation" -n 50 --gemini-model gemini-1.5-pro

    # Evaluate with specific metrics only
    $0 -m ./my-model -s --metrics correctness coherence

    # Evaluate on local dataset
    $0 -m ./my-model -f ./test_data.json

NOTES:
    - Gemini API has rate limits: be mindful with large test sizes
    - Using gemini-1.5-flash is recommended for cost-effectiveness
    - Sample data includes Indonesian conversation examples
    - Results are saved in JSON and Markdown formats
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
        -o|--output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        -d|--dataset)
            DATASET_NAME="$2"
            shift 2
            ;;
        -f|--dataset-path)
            DATASET_PATH="$2"
            shift 2
            ;;
        -s|--use-sample-data)
            USE_SAMPLE_DATA=true
            shift
            ;;
        -n|--test-size)
            TEST_SIZE="$2"
            shift 2
            ;;
        --gemini-model)
            GEMINI_MODEL="$2"
            if [[ ! "$GEMINI_MODEL" =~ ^(gemini-1\.5-pro|gemini-1\.5-flash|gemini-1\.0-pro)$ ]]; then
                print_error "Invalid Gemini model: $GEMINI_MODEL"
                echo "Valid options: gemini-1.5-pro, gemini-1.5-flash, gemini-1.0-pro"
                exit 1
            fi
            shift 2
            ;;
        --metrics)
            shift
            while [[ $# -gt 0 ]] && [[ ! $1 =~ ^- ]]; do
                METRICS="$METRICS $1"
                shift
            done
            ;;
        --no-4bit)
            NO_4BIT=true
            shift
            ;;
        --script)
            EVALUATION_SCRIPT="$2"
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

print_header "Qwen3 Model Evaluation - DeepEval with Google Gemini"

# Check if model path exists
if [ ! -d "$MODEL_PATH" ]; then
    print_error "Model path not found: $MODEL_PATH"
    print_info "Available models in current directory:"
    ls -d SFT-* 2>/dev/null || echo "  No models found"
    exit 1
fi

# Check if evaluation script exists
if [ ! -f "$EVALUATION_SCRIPT" ]; then
    print_error "Evaluation script not found: $EVALUATION_SCRIPT"
    print_info "Please ensure the Gemini-integrated evaluation script is available."
    exit 1
fi

# Validate evaluation mode
if [ "$USE_SAMPLE_DATA" = false ] && [ -z "$DATASET_NAME" ] && [ -z "$DATASET_PATH" ]; then
    print_error "Must specify either --use-sample-data, --dataset, or --dataset-path"
    usage
    exit 1
fi

print_info "Model path: $MODEL_PATH"
print_info "Evaluation script: $EVALUATION_SCRIPT"
print_info "Output directory: $OUTPUT_DIR"
print_info "Gemini model: $GEMINI_MODEL"

if [ "$USE_SAMPLE_DATA" = true ]; then
    print_info "Evaluation mode: Sample data (Indonesian examples)"
elif [ -n "$DATASET_NAME" ]; then
    print_info "Evaluation mode: HuggingFace dataset ($DATASET_NAME)"
    print_info "Test size: $TEST_SIZE samples"
elif [ -n "$DATASET_PATH" ]; then
    print_info "Evaluation mode: Local dataset ($DATASET_PATH)"
    print_info "Test size: $TEST_SIZE samples"
fi

if [ -n "$METRICS" ]; then
    print_info "Using specific metrics:$METRICS"
fi
echo

# Run checks
check_gpu
echo
check_dependencies
echo
check_gemini_key
echo

# Create output directory
mkdir -p "$OUTPUT_DIR"
print_info "Created output directory: $OUTPUT_DIR"
echo

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
except Exception as e:
    print(f"Could not read training info: {e}")
EOF
else
    print_warning "training_info.json not found in model directory"
fi
echo

# Build command
CMD="python $EVALUATION_SCRIPT --model-path \"$MODEL_PATH\" --output-dir \"$OUTPUT_DIR\" --gemini-model \"$GEMINI_MODEL\""

if [ -n "$BASE_MODEL" ]; then
    CMD="$CMD --base-model \"$BASE_MODEL\""
fi

if [ "$USE_SAMPLE_DATA" = true ]; then
    CMD="$CMD --use-sample-data"
fi

if [ -n "$DATASET_NAME" ]; then
    CMD="$CMD --dataset-name \"$DATASET_NAME\" --test-size $TEST_SIZE"
fi

if [ -n "$DATASET_PATH" ]; then
    CMD="$CMD --dataset-path \"$DATASET_PATH\" --test-size $TEST_SIZE"
fi

if [ "$NO_4BIT" = true ]; then
    CMD="$CMD --no-4bit"
fi

if [ -n "$METRICS" ]; then
    CMD="$CMD --metrics$METRICS"
fi

# Display Gemini pricing information
print_header "Gemini API Pricing Information"
echo "Current Gemini pricing (subject to change):"
echo "  - gemini-1.5-flash: ~$0.075 per 1M input tokens, ~$0.30 per 1M output tokens"
echo "  - gemini-1.5-pro: ~$1.25 per 1M input tokens, ~$5.00 per 1M output tokens"
echo "  - gemini-1.0-pro: ~$0.50 per 1M input tokens, ~$1.50 per 1M output tokens"
echo
echo "Estimated cost for this evaluation (rough estimate):"
if [ "$USE_SAMPLE_DATA" = true ]; then
    echo "  Sample data: < $0.01 with gemini-1.5-flash"
else
    echo "  $TEST_SIZE samples: ~$0.01-0.10 with gemini-1.5-flash (depends on text length)"
fi
echo

# Confirm before starting
read -p "$(echo -e ${GREEN}Do you want to start evaluation? [y/N]:${NC} )" -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    print_warning "Evaluation cancelled by user."
    exit 0
fi

# Start evaluation
print_header "Starting Evaluation with Gemini"
print_info "Evaluation started at: $(date)"
print_warning "This may take a while depending on test size, metrics, and Gemini API rate limits..."
echo

# Create timestamp for log file
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="${OUTPUT_DIR}/evaluation_gemini_${TIMESTAMP}.log"

# Run evaluation with logging
eval $CMD 2>&1 | tee "$LOG_FILE"

# Check evaluation result
if [ ${PIPESTATUS[0]} -eq 0 ]; then
    print_header "Evaluation Completed Successfully"
    print_info "Evaluation finished at: $(date)"
    print_info "Results saved to: $OUTPUT_DIR"
    print_info "Log file: $LOG_FILE"
    print_info "Evaluation model: Google Gemini ($GEMINI_MODEL)"
    echo
    
    # List generated files
    print_info "Generated files:"
    ls -lh "$OUTPUT_DIR" | grep -v "^total" | awk '{print "  - " $9 " (" $5 ")"}'
    echo
    
    # Display summary if available
    SUMMARY_FILE="$OUTPUT_DIR/evaluation_summary.json"
    if [ -f "$SUMMARY_FILE" ]; then
        print_header "Evaluation Summary"
        python - <<EOF
import json
try:
    with open("$SUMMARY_FILE", 'r') as f:
        summary = json.load(f)
    
    if 'single_turn' in summary:
        print("\nSingle-Turn Metrics:")
        for metric, value in summary['single_turn'].items():
            if isinstance(value, (int, float)):
                print(f"  - {metric}: {value:.3f}")
    
    if 'multi_turn' in summary:
        print("\nMulti-Turn Metrics:")
        for metric, value in summary['multi_turn'].items():
            if isinstance(value, (int, float)):
                print(f"  - {metric}: {value:.3f}")
    
    print(f"\nEvaluated using: Google Gemini ($GEMINI_MODEL)")
except Exception as e:
    print(f"Could not read summary: {e}")
EOF
    fi
    
    # Check for markdown report
    REPORT_FILE="$OUTPUT_DIR/evaluation_report.md"
    if [ -f "$REPORT_FILE" ]; then
        echo
        print_info "Detailed report available at: $REPORT_FILE"
        print_info "View with: cat $REPORT_FILE"
    fi
else
    print_header "Evaluation Failed"
    print_error "Evaluation failed. Check logs at: $LOG_FILE"
    echo
    print_info "Common issues:"
    echo "  1. Gemini API key not set or invalid"
    echo "  2. Rate limiting from Gemini API"
    echo "  3. Network connectivity issues"
    echo "  4. Model loading errors (insufficient GPU memory)"
    echo
    print_info "To debug, check the log file and ensure:"
    echo "  - GEMINI_API_KEY or GOOGLE_API_KEY is set correctly"
    echo "  - You have sufficient API quota"
    echo "  - The model path is correct"
    exit 1
fi
