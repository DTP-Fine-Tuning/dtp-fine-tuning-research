#!/bin/bash

################################################################################
# Master Script for Qwen3 Fine-tuning Pipeline
# Runs training, inference, and evaluation in sequence or individually
################################################################################

set -e  # Exit on error

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
MAGENTA='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m'

################################################################################
# Helper Functions
################################################################################

print_header() {
    echo
    echo -e "${BLUE}╔════════════════════════════════════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${BLUE}║${NC} ${CYAN}$1${NC}"
    echo -e "${BLUE}╚════════════════════════════════════════════════════════════════════════════════════════════╝${NC}"
    echo
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

print_step() {
    echo
    echo -e "${MAGENTA}>>> $1${NC}"
    echo
}

show_banner() {
    cat << "EOF"
    ╔═══════════════════════════════════════════════════════════════════╗
    ║                                                                   ║
    ║         Qwen3-1.7B Fine-tuning Pipeline                           ║
    ║         Complete Training, Inference & Evaluation                 ║
    ║                                                                   ║
    ╚═══════════════════════════════════════════════════════════════════╝
EOF
}

################################################################################
# Pipeline Operations
################################################################################

run_training() {
    print_header "Step 1: Fine-tuning Training"
    
    if [ -f "run_training.sh" ]; then
        bash run_training.sh "$@"
    else
        print_error "run_training.sh not found"
        return 1
    fi
}

run_inference() {
    print_header "Step 2: Model Inference"
    
    local model_path="$1"
    shift
    
    if [ -f "run_inference.sh" ]; then
        bash run_inference.sh --model-path "$model_path" "$@"
    else
        print_error "run_inference.sh not found"
        return 1
    fi
}

run_evaluation() {
    print_header "Step 3 (OpenAI): Model Evaluation"
    
    local model_path="$1"
    shift
    
    if [ -f "run_evaluation.sh" ]; then
        bash run_evaluation.sh --model-path "$model_path" "$@"
    else
        print_error "run_evaluation.sh not found"
        return 1
    fi
}

run_evaluation_gemini() {
    print_header "Step 3 (Gemini): Model Evaluation"
    
    local model_path="$1"
    shift
    
    if [ -f "run_evaluation_gemini.sh" ]; then
        bash run_evaluation_gemini.sh --model-path "$model_path" "$@"
    else
        print_error "run_evaluation_gemini.sh not found"
        return 1
    fi
}

################################################################################
# Menu System
################################################################################

show_menu() {
    echo
    echo -e "${CYAN}┌─────────────────────────────────────────────────┐${NC}"
    echo -e "${CYAN}│${NC}  What would you like to do?                    ${CYAN}│${NC}"
    echo -e "${CYAN}├─────────────────────────────────────────────────┤${NC}"
    echo -e "${CYAN}│${NC}  1) Run complete pipeline (Train → Eval → Infer)"
    echo -e "${CYAN}│${NC}  2) Run training only                          ${CYAN}│${NC}"
    echo -e "${CYAN}│${NC}  3) Run inference only                         ${CYAN}│${NC}"
    echo -e "${CYAN}│${NC}  4) Run evaluation (OpenAI)                  ${CYAN}│${NC}"
    echo -e "${CYAN}│${NC}  5) Run evaluation (Gemini)                  ${CYAN}│${NC}"
    echo -e "${CYAN}│${NC}  6) Quick test (OpenAI)                      ${CYAN}│${NC}"
    echo -e "${CYAN}│${NC}  7) Quick test (Gemini)                      ${CYAN}│${NC}"
    echo -e "${CYAN}│${NC}  8) Exit                                       ${CYAN}│${NC}"
    echo -e "${CYAN}└─────────────────────────────────────────────────┘${NC}"
    echo
}

get_model_path() {
    local default_path="./SFT-Qwen3-1.7B-LoRA-MultiTurn-final"
    
    echo
    print_info "Available models:"
    ls -d SFT-* 2>/dev/null || echo "  No models found"
    echo
    
    read -p "Enter model path [${default_path}]: " model_path
    model_path=${model_path:-$default_path}
    
    if [ ! -d "$model_path" ]; then
        print_error "Model path not found: $model_path"
        return 1
    fi
    
    echo "$model_path"
}

################################################################################
# Pipeline Modes
################################################################################

complete_pipeline() {
    print_header "Running Complete Pipeline"
    
    print_step "Phase 1/3: Training"
    if ! run_training; then
        print_error "Training failed. Stopping pipeline."
        return 1
    fi
    
    # Get the output directory from training
    local config_file="configs/sft_qwen3_1_7B_improved.yaml"
    local model_path=$(python -c "import yaml; config = yaml.safe_load(open('$config_file')); print(config['paths']['final_model_dir'])" 2>/dev/null)
    
    if [ -z "$model_path" ] || [ ! -d "$model_path" ]; then
        print_error "Could not find trained model. Using default path."
        model_path="./SFT-Qwen3-1.7B-LoRA-MultiTurn-final"
    fi
    
    print_step "Phase 2/3: Evaluation (Quick Test - OpenAI)"
    if ! run_evaluation "$model_path" --use-sample-data; then
        print_warning "Evaluation failed, but continuing to inference."
    fi
    
    print_step "Phase 3/3: Inference"
    read -p "Do you want to launch Gradio interface? [y/N]: " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        run_inference "$model_path"
    fi
    
    print_header "Pipeline Completed"
    print_info "Model location: $model_path"
}

quick_test() {
    print_header "Quick Test Mode (OpenAI)"
    
    local model_path
    if ! model_path=$(get_model_path); then
        return 1
    fi
    
    print_info "Running quick evaluation with sample data (OpenAI)..."
    run_evaluation "$model_path" --use-sample-data
}

# FUNGSI BARU DITAMBAHKAN
quick_test_gemini() {
    print_header "Quick Test Mode (Gemini)"
    
    local model_path
    if ! model_path=$(get_model_path); then
        return 1
    fi
    
    print_info "Running quick evaluation with sample data (Gemini)..."
    run_evaluation_gemini "$model_path" --use-sample-data
}

################################################################################
# Main Menu Loop
################################################################################

main_menu() {
    show_banner
    
    while true; do
        show_menu
        read -p "Select option [1-8]: " choice
        
        case $choice in
            1)
                complete_pipeline
                ;;
            2)
                run_training
                ;;
            3)
                local model_path
                if model_path=$(get_model_path); then
                    run_inference "$model_path"
                fi
                ;;
            4)
                local model_path
                if model_path=$(get_model_path); then
                    run_evaluation "$model_path"
                fi
                ;;
            5) # <-- OPSI BARU
                local model_path
                if model_path=$(get_model_path); then
                    run_evaluation_gemini "$model_path"
                fi
                ;;
            6)
                quick_test
                ;;
            7) # <-- OPSI BARU
                quick_test_gemini
                ;;
            8)
                print_info "Exiting..."
                exit 0
                ;;
            *)
                print_error "Invalid option. Please select 1-8."
                ;;
        esac
        
        echo
        read -p "Press Enter to continue..." dummy
    done
}

################################################################################
# Command Line Mode
################################################################################

usage() {
    cat << EOF
Usage: $0 [MODE] [OPTIONS]

MODES:
    pipeline                Run complete pipeline (train → eval → inference)
    train                   Run training only
    inference               Run inference only
    evaluate                Run evaluation only (OpenAI)
    evaluate-gemini         Run evaluation only (Gemini)
    quick-test              Quick evaluation with sample data (OpenAI)
    quick-test-gemini       Quick evaluation with sample data (Gemini)
    menu                    Show interactive menu (default)

OPTIONS (for specific modes):
    For training:
        -c, --config FILE   Configuration file
    
    For inference:
        -m, --model PATH    Model path
        -p, --port PORT     Server port
        -s, --share         Create public link
    
    For evaluation (OpenAI & Gemini):
        -m, --model PATH    Model path
        -d, --dataset NAME  Dataset name
        -s, --sample        Use sample data

EXAMPLES:
    # Interactive menu (default)
    $0

    # Run complete pipeline
    $0 pipeline

    # Run training with custom config
    $0 train -c configs/my_config.yaml

    # Run inference on specific model
    $0 inference -m ./my-model -p 8080 -s

    # Quick evaluation test (Gemini)
    $0 quick-test-gemini -m ./my-model

    # Full evaluation on dataset (OpenAI)
    $0 evaluate -m ./my-model -d "izzulgod/indonesian-conversation"
EOF
}

################################################################################
# Main Entry Point
################################################################################

# Check if any arguments provided
if [ $# -eq 0 ]; then
    # No arguments, show menu
    main_menu
else
    MODE="$1"
    shift
    
    case $MODE in
        pipeline|complete)
            complete_pipeline "$@"
            ;;
        train|training)
            run_training "$@"
            ;;
        inference|infer)
            # Extract model path from arguments
            model_path=""
            while [[ $# -gt 0 ]]; do
                case $1 in
                    -m|--model|--model-path)
                        model_path="$2"
                        shift 2
                        ;;
                    *)
                        break
                        ;;
                esac
            done
            
            if [ -z "$model_path" ]; then
                model_path=$(get_model_path) || exit 1
            fi
            
            run_inference "$model_path" "$@"
            ;;
        eval|evaluate|evaluation)
            # Extract model path from arguments
            model_path=""
            while [[ $# -gt 0 ]]; do
                case $1 in
                    -m|--model|--model-path)
                        model_path="$2"
                        shift 2
                        ;;
                    *)
                        break
                        ;;
                esac
            done
            
            if [ -z "$model_path" ]; then
                model_path=$(get_model_path) || exit 1
            fi
            
            run_evaluation "$model_path" "$@"
            ;;
        
        evaluate-gemini|eval-gemini)
            # Extract model path from arguments
            model_path=""
            while [[ $# -gt 0 ]]; do
                case $1 in
                    -m|--model|--model-path)
                        model_path="$2"
                        shift 2
                        ;;
                    *)
                        break
                        ;;
                esac
            done
            
            if [ -z "$model_path" ]; then
                model_path=$(get_model_path) || exit 1
            fi
            
            run_evaluation_gemini "$model_path" "$@"
            ;;

        quick-test|test)
            if [ $# -gt 0 ] && [[ "$1" =~ ^(-m|--model) ]]; then
                model_path="$2"
            else
                model_path=$(get_model_path) || exit 1
            fi
            
            run_evaluation "$model_path" --use-sample-data
            ;;
        
        quick-test-gemini|test-gemini)
            if [ $# -gt 0 ] && [[ "$1" =~ ^(-m|--model) ]]; then
                model_path="$2"
            else
                model_path=$(get_model_path) || exit 1
            fi
            
            run_evaluation_gemini "$model_path" --use-sample-data
            ;;

        menu)
            main_menu
            ;;
        -h|--help|help)
            usage
            exit 0
            ;;
        *)
            print_error "Unknown mode: $MODE"
            usage
            exit 1
            ;;
    esac
fi