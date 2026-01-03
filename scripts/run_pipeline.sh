#!/bin/bash
#master script to run the complete fine-tuning pipeline:
# training -> evaluation -> inference
#author: Tim 2 DTP

set -e  

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
MAGENTA='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m'

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
    ║         Fine-tuning Pipeline                                      ║
    ║         Complete Training, Inference & Evaluation                 ║
    ║                                                                   ║
    ╚═══════════════════════════════════════════════════════════════════╝
EOF
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
        print_info "Please cd to project root and run: ./scripts/run_pipeline.sh"
        exit 1
    fi
}

run_training() {
    print_header "Step 1: Fine-tuning Training"
    
    if [ -f "scripts/run_training_unsloth.sh" ]; then
        bash scripts/run_training_unsloth.sh "$@"
    else
        print_error "scripts/run_training_unsloth.sh not found"
        return 1
    fi
}


run_evaluation() {
    print_header "Step 2: Model Evaluation"
    
    local model_path="$1"
    shift
    
    if [ -f "scripts/run_evaluation.sh" ]; then
        bash scripts/run_evaluation.sh --model-path "$model_path" "$@"
    else
        print_error "scripts/run_evaluation.sh not found"
        return 1
    fi
}

run_inference() {
    print_header "Step 3: Model Inference"
    
    local model_path="$1"
    shift
    
    if [ -f "scripts/run_inference.sh" ]; then
        bash scripts/run_inference.sh --model-path "$model_path" "$@"
    else
        print_error "scripts/run_inference.sh not found"
        return 1
    fi
}

show_menu() {
    echo
    echo -e "${CYAN}┌─────────────────────────────────────────────────┐${NC}"
    echo -e "${CYAN}│${NC}  What would you like to do?                    ${CYAN}│${NC}"
    echo -e "${CYAN}├─────────────────────────────────────────────────┤${NC}"
    echo -e "${CYAN}│${NC}  1) Run complete pipeline (Train → Eval → Infer) ${CYAN}│${NC}"
    echo -e "${CYAN}│${NC}  2) Run training only                          ${CYAN}│${NC}"
    echo -e "${CYAN}│${NC}  3) Run inference only                         ${CYAN}│${NC}"
    echo -e "${CYAN}│${NC}  4) Run evaluation (deepeval)                    ${CYAN}│${NC}"
    echo -e "${CYAN}│${NC}  5) Quick test (openrouter)                        ${CYAN}│${NC}"
    echo -e "${CYAN}│${NC}  6) Exit                                       ${CYAN}│${NC}"
    echo -e "${CYAN}└─────────────────────────────────────────────────┘${NC}"
    echo
}

get_model_path() {
    local default_path=""
    local latest_model=""
    local latest_time=0
    
    if compgen -G "SFT-*" > /dev/null; then
        for model in SFT-*/; do
            if [ -d "$model" ]; then
                model_time=$(stat -c %Y "$model" 2>/dev/null || stat -f %m "$model" 2>/dev/null || echo 0)
                if [ "$model_time" -gt "$latest_time" ]; then
                    latest_time=$model_time
                    latest_model=$model
                fi
            fi
        done
    fi
    
    if compgen -G "SFT-*" > /dev/null; then
        for model in SFT-*/; do
            if [ -d "$model" ]; then
                model_time=$(stat -c %Y "$model" 2>/dev/null || stat -f %m "$model" 2>/dev/null || echo 0)
                if [ "$model_time" -gt "$latest_time" ]; then
                    latest_time=$model_time
                    latest_model=$model
                fi
            fi
        done
    fi
    
    if [ -n "$latest_model" ]; then
        default_path="$latest_model"
    fi
    
    echo
    print_info "Available models:"
    
    if compgen -G "SFT-*" > /dev/null; then
        for model in SFT-*/; do
            if [ -d "$model" ]; then
                echo "  - $model"
            fi
        done
    fi
    
    if compgen -G "SFT-*" > /dev/null; then
        for model in SFT-*/; do
            if [ -d "$model" ]; then
                echo "  - $model"
            fi
        done
    fi
    
    if [ -z "$default_path" ]; then
        echo "  No models found"
        print_error "No models found. Please run training first."
        return 1
    fi
    
    echo
    
    read -p "Enter model path [${default_path}]: " model_path
    model_path=${model_path:-$default_path}
    
    if [ ! -d "$model_path" ]; then
        print_error "Model path not found: $model_path"
        return 1
    fi
    
    echo "$model_path"
}

complete_pipeline() {
    print_header "Running Complete Pipeline"
    
    local config_file=""
    local use_sample_data=true
    local scenario_file=""
    local share_gradio=false
    
    # Parse pipeline-specific options
    while [[ $# -gt 0 ]]; do
        case $1 in
            -c|--config)
                config_file="$2"
                shift 2
                ;;
            --scenario-file)
                scenario_file="$2"
                shift 2
                ;;
            --full-eval)
                use_sample_data=false
                shift
                ;;
            --share)
                share_gradio=true
                shift
                ;;
            *)
                shift
                ;;
        esac
    done
    
    # Interactive config selection if not provided
    if [ -z "$config_file" ]; then
        echo
        print_info "Available training configurations:"
        local count=0
        local configs=()
        for conf in configs/*.yaml configs/test/*.yaml; do
            if [ -f "$conf" ]; then
                configs+=("$conf")
                echo "  [$count] $conf"
                ((count++))
            fi
        done
        echo
        read -p "Select config [number] or press Enter for default [configs/test/sft_diploy_8B.yaml]: " config_choice
        
        if [ -z "$config_choice" ]; then
            config_file="configs/test/sft_diploy_8B.yaml"
        elif [[ "$config_choice" =~ ^[0-9]+$ ]] && [ "$config_choice" -lt "${#configs[@]}" ]; then
            config_file="${configs[$config_choice]}"
        else
            config_file="$config_choice"
        fi
        print_info "Using config: $config_file"
    fi
    
    if [ ! -f "$config_file" ]; then
        print_error "Config file not found: $config_file"
        return 1
    fi
    
    print_step "Phase 1/3: Training"
    if ! run_training -c "$config_file"; then
        print_error "Training failed. Stopping pipeline."
        return 1
    fi
    
    local model_path=$(python -c "import yaml; config = yaml.safe_load(open('$config_file')); print(config.get('paths', {}).get('final_model_dir', config['training']['output_dir']))" 2>/dev/null)
    
    if [ -z "$model_path" ] || [ ! -d "$model_path" ]; then
        print_error "Could not find trained model. Searching for latest model..."
        model_path=$(get_model_path) || return 1
    fi
    
    print_step "Phase 2/3: Evaluation (deepeval)"
    
    # Build evaluation command
    local eval_args=("$model_path")
    if [ "$use_sample_data" = true ]; then
        echo
        read -p "Use sample data for quick evaluation? [Y/n]: " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Nn]$ ]]; then
            eval_args+=("--use-sample-data")
        fi
    fi
    
    if [ -n "$scenario_file" ]; then
        eval_args+=("--scenario-file" "$scenario_file")
    fi
    
    if ! run_evaluation "${eval_args[@]}"; then
        print_warning "Evaluation failed, but continuing to inference."
    fi
    
    print_step "Phase 3/3: Inference"
    echo
    read -p "Do you want to launch Gradio interface? [y/N]: " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        local inference_args=("$model_path")
        
        if [ "$share_gradio" = false ]; then
            read -p "Create public Gradio link (--share)? [y/N]: " -n 1 -r
            echo
            if [[ $REPLY =~ ^[Yy]$ ]]; then
                inference_args+=("--share")
            fi
        else
            inference_args+=("--share")
        fi
        
        run_inference "${inference_args[@]}"
    fi
    
    print_header "Pipeline Completed"
    print_info "Model location: $model_path"
    print_info "Config used: $config_file"
}

quick_test() {
    print_header "Quick Test Mode (deepeval)"
    
    local model_path
    if ! model_path=$(get_model_path); then
        return 1
    fi
    
    print_info "Running quick evaluation with sample data (deepeval)..."
    run_evaluation "$model_path" --use-sample-data
}

main_menu() {
    show_banner
    
    while true; do
        show_menu
        read -p "Select option [1-6]: " choice
        
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
            5)
                quick_test
                ;;
            6)
                print_info "Exiting..."
                exit 0
                ;;
            *)
                print_error "Invalid option. Please select 1-6."
                ;;
        esac
        
        echo
        read -p "Press Enter to continue..." dummy
    done
}

usage() {
    cat << EOF
Usage: $0 [MODE] [OPTIONS]

MODES:
    pipeline                Run complete pipeline (train → eval → inference)
    train                   Run training only
    inference               Run inference only
    evaluate                Run evaluation only (OpenAI)
    quick-test              Quick evaluation with sample data (OpenAI)
    menu                    Show interactive menu (default)

OPTIONS (for specific modes):
    For pipeline:
        -c, --config FILE       Training configuration file
        --scenario-file FILE    Custom evaluation scenarios (JSON)
        --share                 Create public Gradio link for inference
        --full-eval             Run full evaluation (not sample data)
    
    For training:
        -c, --config FILE       Configuration file
    
    For inference:
        -m, --model PATH        Model path
        -p, --port PORT         Server port
        -s, --share             Create public link
    
    For evaluation:
        -m, --model PATH        Model path
        --scenario-file FILE    Custom scenario JSON file
        --use-sample-data       Use default sample scenarios
        -d, --dataset NAME      Dataset name
        -s, --sample            Use sample data

EXAMPLES:
    # Interactive menu (default)
    ./scripts/run_pipeline.sh

    # Run complete pipeline with custom config
    ./scripts/run_pipeline.sh pipeline -c configs/my_config.yaml

    # Run complete pipeline with custom scenarios and public sharing
    ./scripts/run_pipeline.sh pipeline -c configs/my_config.yaml --scenario-file scenarios.json --share

    # Run training with custom config
    ./scripts/run_pipeline.sh train -c configs/my_config.yaml

    # Run inference on specific model with public link
    ./scripts/run_pipeline.sh inference -m src/training/SFT-Qwen3-1.7B-LoRA-9GB-final --share

    # Full evaluation with custom scenarios
    ./scripts/run_pipeline.sh evaluate -m model_path --scenario-file my_scenarios.json
EOF
}

verify_project_root
load_env_file

if [ $# -eq 0 ]; then
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

        quick-test|test)
            if [ $# -gt 0 ] && [[ "$1" =~ ^(-m|--model) ]]; then
                model_path="$2"
            else
                model_path=$(get_model_path) || exit 1
            fi
            
            run_evaluation "$model_path" --use-sample-data
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