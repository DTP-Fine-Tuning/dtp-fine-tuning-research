#!/bin/bash
# deepeval scripts - Interactive Conversational Evaluation
# author: Tim 2 DTP
# adapted for deepeval_my_model.py approach

set -e
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

MODEL_PATH=""
EVALUATION_SCRIPT="src/eval/deepeval_my_model.py"
OUTPUT_DIR="evaluation_results"
JUDGE_MODEL=""
OPENROUTER_API_KEY=""
LOAD_IN_4BIT=true
GENERATION_TEMPERATURE=0.3
NUM_SCENARIOS=3
SCENARIO_FILE=""
USE_SAMPLE_DATA=false

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

print_highlight() {
    echo -e "${CYAN}[SELECT]${NC} $1"
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
        print_info "Please cd to project root and run: ./scripts/run_evaluation.sh"
        exit 1
    fi
}

find_latest_model() {
    local latest_model=""
    local latest_time=0
    
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
    print_info "Available models in workspace:"
    local count=0
    local models=()
    
    if compgen -G "src/training/SFT-*" > /dev/null; then
        for model in src/training/SFT-*/; do
            if [ -d "$model" ]; then
                models+=("$model")
                echo "  [$count] $model"
                ((count++))
            fi
        done
    fi
    
    if compgen -G "src/utils/SFT-*" > /dev/null; then
        for model in src/utils/SFT-*/; do
            if [ -d "$model" ]; then
                models+=("$model")
                echo "  [$count] $model"
                ((count++))
            fi
        done
    fi
    
    if compgen -G "sft_*" > /dev/null; then
        for model in sft_*/; do
            if [ -d "$model" ]; then
                models+=("$model")
                echo "  [$count] $model"
                ((count++))
            fi
        done
    fi
    
    echo ""
    print_info "Or enter a HuggingFace model path (e.g., wildanaziz/Diploy-8B-Base)"
    
    if [ $count -eq 0 ]; then
        echo "  No local models found. Please use HuggingFace model or train a model first."
        return 1
    fi
    
    echo "${models[@]}"
    return 0
}

interactive_model_selection() {
    print_header "Model Selection"
    
    local models_output=$(list_available_models)
    local models=($(echo "$models_output" | tail -1))
    
    echo ""
    print_highlight "Enter model selection:"
    read -p "  [Number] for local model, [Path] for HuggingFace model, or [Enter] for latest: " model_choice
    
    if [ -z "$model_choice" ]; then
        MODEL_PATH=$(find_latest_model)
        if [ -z "$MODEL_PATH" ]; then
            print_error "No models found"
            exit 1
        fi
        print_info "Using latest model: $MODEL_PATH"
    elif [[ "$model_choice" =~ ^[0-9]+$ ]]; then
        if [ "$model_choice" -lt "${#models[@]}" ]; then
            MODEL_PATH="${models[$model_choice]}"
            print_info "Selected local model: $MODEL_PATH"
        else
            print_error "Invalid selection number"
            exit 1
        fi
    else
        MODEL_PATH="$model_choice"
        print_info "Using custom model path: $MODEL_PATH"
    fi
}

select_judge_model() {
    print_header "Judge Model Selection (OpenRouter API)"
    
    echo ""
    print_info "Available judge models:"
    echo "  [1] openai/gpt-4o-mini (Recommended - Best balance)"
    echo "  [2] openai/gpt-3.5-turbo (Faster, cheaper)"
    echo "  [3] openai/gpt-4o (Most accurate, expensive)"
    echo "  [4] anthropic/claude-3.5-sonnet (Alternative, high quality)"
    echo "  [5] anthropic/claude-3-haiku (Fast, affordable)"
    echo "  [6] Custom model ID"
    
    echo ""
    print_highlight "Select judge model [1-6] (default: 1):"
    read -p "  Your choice: " judge_choice
    
    case ${judge_choice:-1} in
        1)
            JUDGE_MODEL="openai/gpt-4o-mini"
            print_info "Using: GPT-4o Mini (balanced)"
            ;;
        2)
            JUDGE_MODEL="openai/gpt-3.5-turbo"
            print_info "Using: GPT-3.5 Turbo (fast)"
            ;;
        3)
            JUDGE_MODEL="openai/gpt-4o"
            print_info "Using: GPT-4o (most accurate)"
            ;;
        4)
            JUDGE_MODEL="anthropic/claude-3.5-sonnet"
            print_info "Using: Claude 3.5 Sonnet"
            ;;
        5)
            JUDGE_MODEL="anthropic/claude-3-haiku"
            print_info "Using: Claude 3 Haiku"
            ;;
        6)
            read -p "  Enter custom model ID: " custom_model
            JUDGE_MODEL="$custom_model"
            print_info "Using: $JUDGE_MODEL"
            ;;
        *)
            JUDGE_MODEL="openai/gpt-4o-mini"
            print_warning "Invalid choice, using default: GPT-4o Mini"
            ;;
    esac
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
    
    python -c "import torch; import transformers; import peft; import deepeval; import datasets" 2>/dev/null
    if [ $? -ne 0 ]; then
        print_warning "Some required packages are missing. Installing dependencies..."
        pip install torch transformers peft deepeval datasets pandas
    else
        print_info "All required packages are installed."
    fi
}

check_deepeval_login() {
    print_info "Checking DeepEval configuration..."
    
    if [ ! -f "$HOME/.deepeval/config.json" ] && [ ! -f ".deepeval/config.json" ]; then
        print_warning "DeepEval not configured yet."
        print_info "DeepEval requires login for tracking evaluation results."
        echo ""
        print_highlight "Running 'deepeval login' to configure..."
        echo ""
        
        deepeval login
        
        if [ $? -ne 0 ]; then
            print_error "DeepEval login failed. Evaluation may not track results properly."
            read -p "Continue anyway? [y/N]: " -n 1 -r
            echo
            if [[ ! $REPLY =~ ^[Yy]$ ]]; then
                exit 1
            fi
        else
            print_info "DeepEval login successful!"
        fi
    else
        print_info "DeepEval already configured."
    fi
}

check_openai_key() {
    if [ -z "$OPENROUTER_API_KEY" ]; then
        print_warning "OPENROUTER_API_KEY not set in environment."
        print_info "DeepEval uses OpenRouter API for evaluation metrics as judge."
        echo ""
        print_highlight "Enter your OpenRouter API key (or press Enter to use default):"
        read -s -p "  API Key: " user_api_key
        echo ""
        
        if [ -n "$user_api_key" ]; then
            OPENROUTER_API_KEY="$user_api_key"
            export OPENROUTER_API_KEY
            print_info "OpenRouter API key set successfully."
        fi
    else
        print_info "OpenRouter API key found in environment."
    fi
}

usage() {
    cat << EOF
Usage: $0 [OPTIONS]

DESCRIPTION:
    Interactive conversational evaluation script using DeepEval framework.
    Evaluates chatbot models with conversational and safety metrics.
    Adapted from deepeval_my_model.py approach.

OPTIONS:
    -m, --model-path PATH       Path to fine-tuned model (interactive if not specified)
    -j, --judge-model MODEL     Judge model for evaluation (interactive if not specified)
    -k, --api-key KEY           OpenRouter API key (interactive if not specified)
    -o, --output-dir DIR        Output directory for results (default: $OUTPUT_DIR)
    -s, --script FILE           Evaluation script path (default: $EVALUATION_SCRIPT)
    --scenario-file FILE        Path to custom scenario JSON file
    --use-sample-data           Use default sample scenarios (quick test)
    --no-4bit                   Disable 4-bit quantization for model loading
    --temperature TEMP          Generation temperature (default: $GENERATION_TEMPERATURE)
    --scenarios NUM             Number of test scenarios (default: $NUM_SCENARIOS)
    -h, --help                  Show this help message

EVALUATION METRICS:
    Conversational Metrics (5):
      - Turn Relevancy: Response addresses user message
      - Knowledge Retention: Remembers context across turns
      - Role Adherence: Maintains professional interviewer persona
      - Conversation Completeness: Gathers all needed information
      - Topic Adherence: Stays focused on relevant topics
    
    Safety Metrics (3):
      - Toxicity: Detects harmful/offensive language
      - Bias: Identifies discrimination
      - Hallucination: Ensures factual accuracy

EXAMPLES:
    # Interactive mode (recommended)
    ./scripts/run_evaluation.sh

    # Quick test with default scenarios
    ./scripts/run_evaluation.sh -m wildanaziz/Diploy-8B-Base --use-sample-data

    # Full evaluation with custom scenarios
    ./scripts/run_evaluation.sh -m wildanaziz/Diploy-8B-Base --scenario-file my_scenarios.json

    # Full specification
    ./scripts/run_evaluation.sh -m wildanaziz/Diploy-8B-Base -j openai/gpt-4o-mini --scenario-file scenarios.json

    # Disable quantization for larger GPUs
    ./scripts/run_evaluation.sh --no-4bit

NOTES:
    - Evaluation generates conversations dynamically using your model
    - OpenRouter API acts as the judge to evaluate response quality
    - Results include scores for all 8 metrics (conversational + safety)
    - Default judge model: openai/gpt-4o-mini (best balance of quality/cost)
EOF
}

while [[ $# -gt 0 ]]; do
    case $1 in
        -m|--model-path)
            MODEL_PATH="$2"
            shift 2
            ;;
        -j|--judge-model)
            JUDGE_MODEL="$2"
            shift 2
            ;;
        -k|--api-key)
            OPENROUTER_API_KEY="$2"
            shift 2
            ;;
        -o|--output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        -s|--script)
            EVALUATION_SCRIPT="$2"
            shift 2
            ;;
        --no-4bit)
            LOAD_IN_4BIT=false
            shift
            ;;
        --temperature)
            GENERATION_TEMPERATURE="$2"
            shift 2
            ;;
        --scenarios)
            NUM_SCENARIOS="$2"
            shift 2
            ;;
        --scenario-file)
            SCENARIO_FILE="$2"
            shift 2
            ;;
        --use-sample-data)
            USE_SAMPLE_DATA=true
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

print_header "Conversational Model Evaluation - DeepEval"
verify_project_root
load_env_file

if [ -z "$MODEL_PATH" ]; then
    interactive_model_selection
fi

MODEL_PATH="${MODEL_PATH%/}"
if [[ ! "$MODEL_PATH" =~ / ]]; then
    print_error "Invalid model path: $MODEL_PATH"
    exit 1
fi

if [[ "$MODEL_PATH" =~ ^[a-zA-Z0-9_-]+/[a-zA-Z0-9_.-]+$ ]]; then
    print_info "Detected HuggingFace model: $MODEL_PATH"
elif [ ! -d "$MODEL_PATH" ]; then
    print_error "Local model path not found: $MODEL_PATH"
    exit 1
fi

if [ ! -f "$EVALUATION_SCRIPT" ]; then
    print_error "Evaluation script not found: $EVALUATION_SCRIPT"
    print_info "Expected location: src/eval/deepeval_my_model.py"
    exit 1
fi

if [ -z "$JUDGE_MODEL" ]; then
    select_judge_model
fi

check_openai_key

echo ""
print_info "Configuration Summary:"
print_info "  Model to evaluate: $MODEL_PATH"
print_info "  Judge model: $JUDGE_MODEL"
print_info "  Evaluation script: $EVALUATION_SCRIPT"
print_info "  Output directory: $OUTPUT_DIR"
print_info "  4-bit quantization: $LOAD_IN_4BIT"
print_info "  Generation temperature: $GENERATION_TEMPERATURE"
print_info "  Number of scenarios: $NUM_SCENARIOS"
if [ -n "$SCENARIO_FILE" ]; then
    print_info "  Custom scenario file: $SCENARIO_FILE"
else
    print_info "  Using: Default scenarios"
fi
if [ "$USE_SAMPLE_DATA" = true ]; then
    print_info "  Mode: Quick test (sample data)"
else
    print_info "  Mode: Full evaluation"
fi
echo

check_gpu
echo
check_dependencies
echo
check_deepeval_login
echo

mkdir -p "$OUTPUT_DIR"
print_info "Created output directory: $OUTPUT_DIR"
echo

print_header "Evaluation Information"
print_info "This evaluation will:"
print_info "  1. Load your model: $MODEL_PATH"
print_info "  2. Generate $NUM_SCENARIOS interview conversations dynamically"
print_info "  3. Evaluate with 8 metrics (5 conversational + 3 safety)"
print_info "  4. Use $JUDGE_MODEL as the judge/evaluator"
echo
print_info "Metrics to be evaluated:"
print_info "  [Conversational] Turn Relevancy, Knowledge Retention"
print_info "  [Role & Task] Role Adherence, Conversation Completeness, Topic Adherence"
print_info "  [Safety] Toxicity, Bias, Hallucination"
echo

#inform user before starting
read -p "$(echo -e ${GREEN}Start evaluation? [y/N]:${NC} )" -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    print_warning "Evaluation cancelled by user."
    exit 0
fi

#start eval
print_header "Starting Evaluation"
print_info "Evaluation started at: $(date)"
print_warning "This may take 10-30 minutes depending on model size and scenarios..."
echo

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="${OUTPUT_DIR}/evaluation_${TIMESTAMP}.log"

export EVAL_MODEL_PATH="$MODEL_PATH"
export EVAL_JUDGE_MODEL="$JUDGE_MODEL"
export OPENROUTER_API_KEY="$OPENROUTER_API_KEY"
export EVAL_LOAD_4BIT="$LOAD_IN_4BIT"
export EVAL_TEMPERATURE="$GENERATION_TEMPERATURE"
export EVAL_NUM_SCENARIOS="$NUM_SCENARIOS"
export EVAL_SCENARIO_FILE="$SCENARIO_FILE"
export EVAL_USE_SAMPLE_DATA="$USE_SAMPLE_DATA"

#run eval
python "$EVALUATION_SCRIPT" 2>&1 | tee "$LOG_FILE"

EVAL_EXIT_CODE=${PIPESTATUS[0]}

if [ $EVAL_EXIT_CODE -eq 0 ]; then
    print_header "Evaluation Completed Successfully"
    print_info "Evaluation finished at: $(date)"
    print_info "Log file: $LOG_FILE"
    echo
    
    print_info "Evaluation completed with all metrics:"
    print_info "  [DONE] Conversational metrics (5): Turn Relevancy, Knowledge Retention,"
    print_info "    Role Adherence, Conversation Completeness, Topic Adherence"
    print_info "  [DONE] Safety metrics (3): Toxicity, Bias, Hallucination"
    echo
    
    print_info "Check the log file for detailed results and scores."
    print_info "DeepEval results are also available in the deepeval results portal."
else
    print_header "Evaluation Failed"
    print_error "Evaluation failed with exit code: $EVAL_EXIT_CODE"
    print_error "Check logs at: $LOG_FILE"
    exit 1
fi