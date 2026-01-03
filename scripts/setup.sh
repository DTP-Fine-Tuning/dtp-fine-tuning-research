#!/bin/bash
# setup script for first clone project
# author: Tim 2 DTP

set -e
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

print_header() {
    echo
    echo -e "${BLUE}╔════════════════════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${BLUE}║${NC}  $1"
    echo -e "${BLUE}╚════════════════════════════════════════════════════════════════════════════╝${NC}"
    echo
}

print_info() {
    echo -e "${GREEN}[DONE]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

print_error() {
    echo -e "${RED}[FAIL]${NC} $1"
}

print_step() {
    echo
    echo -e "${BLUE}[STEP]${NC} $1"
}

load_env_file() {
    if [ -f ".env" ]; then
        print_info "Found .env file, loading environment variables..."
        set -a  
        source .env
        set +a 
    
        print_info "Environment variables loaded from .env"
    else
        print_warning ".env file not found"
        print_info "You can create one from the template later"
    fi
}

verify_project_root() {
    if [ ! -d "scripts" ] || [ ! -d "src" ] || [ ! -d "configs" ]; then
        print_error "This script must be run from the project root directory"
        print_error "Expected directory structure:"
        echo "  dtp-fine-tuning-research/"
        echo "  ├── scripts/"
        echo "  ├── src/"
        echo "  ├── configs/"
        echo "  └── ..."
        echo
        print_info "Current directory: $(pwd)"
        print_info "Please cd to project root and run: ./scripts/setup.sh"
        exit 1
    fi
    
    print_info "Running from project root: $(pwd)"
}

print_header "Fine-tuning Pipeline Setup"
verify_project_root
load_env_file

print_step "Step 1: Verifying project directory structure..."
REQUIRED_DIRS=(
    "configs"
    "scripts"
    "src"
    "src/training"
    "src/utils"
)
OPTIONAL_DIRS=(
    "logs"
    "evaluation_results"
    "checkpoints"
)

MISSING_REQUIRED=false
for dir in "${REQUIRED_DIRS[@]}"; do
    if [ -d "$dir" ]; then
        print_info "Directory exists: $dir/"
    else
        print_error "Required directory missing: $dir/"
        MISSING_REQUIRED=true
    fi
done

if [ "$MISSING_REQUIRED" = true ]; then
    print_error "Project structure is incomplete"
    exit 1
fi

for dir in "${OPTIONAL_DIRS[@]}"; do
    if [ -d "$dir" ]; then
        print_info "Directory exists: $dir/"
    else
        mkdir -p "$dir"
        print_info "Created directory: $dir/"
    fi
done


print_step "Step 2: Making scripts executable..."
scripts=(
    "scripts/run_pipeline.sh"
    "scripts/run_training_unsloth.sh"
    "scripts/run_inference.sh"
    "scripts/run_evaluation.sh"
    "scripts/setup.sh"
)

for script in "${scripts[@]}"; do
    if [ -f "$script" ]; then
        chmod +x "$script"
        print_info "Made $script executable"
    else
        print_warning "$script not found (this may be expected)"
    fi
done


print_step "Step 3: Checking Python installation.."
if ! command -v python &> /dev/null; then
    print_error "Python not found. Please install Python 3.8 or higher."
    exit 1
fi

PYTHON_VERSION=$(python --version 2>&1 | cut -d' ' -f2)
print_info "Python version: $PYTHON_VERSION"

REQUIRED_VERSION="3.8.0"
if ! python -c "import sys; exit(0 if sys.version_info >= tuple(map(int, '$REQUIRED_VERSION'.split('.'))) else 1)" 2>/dev/null; then
    print_error "Python 3.8 or higher is required"
    exit 1
fi

print_step "Step 4: Checking GPU/CUDA availability..."
if command -v nvidia-smi &> /dev/null; then
    print_info "NVIDIA GPU detected"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | while read line; do
        print_info "  $line"
    done
    
    if python -c "import torch; print(torch.cuda.is_available())" 2>/dev/null | grep -q "True"; then
        print_info "CUDA is available in PyTorch"
        CUDA_VERSION=$(python -c "import torch; print(torch.version.cuda)" 2>/dev/null)
        print_info "CUDA version: $CUDA_VERSION"
    else
        print_warning "CUDA not available in PyTorch"
        print_warning "Training will run on CPU (very slow)"
    fi
else
    print_warning "nvidia-smi not found"
    print_warning "GPU acceleration may not be available"
fi

print_step "Step 5: Checking Python dependencies..."
REQUIRED_PACKAGES_INSTALL=(
    "torch"
    "transformers"
    "peft"
    "trl"
    "datasets"
    "accelerate"
    "bitsandbytes"
    "gradio"
    "deepeval"
    "wandb"
    "pyyaml"
    "google-generativeai"
    "tenacity"
    "unsloth"
)

REQUIRED_PACKAGES_IMPORT=(
    "torch"
    "transformers"
    "peft"
    "trl"
    "datasets"
    "accelerate"
    "bitsandbytes"
    "gradio"
    "deepeval"
    "wandb"
    "yaml"
    "google.generativeai"
    "tenacity"
    "unsloth"
)

MISSING_PACKAGES=()
for i in "${!REQUIRED_PACKAGES_INSTALL[@]}"; do
    package_install="${REQUIRED_PACKAGES_INSTALL[$i]}"
    package_import="${REQUIRED_PACKAGES_IMPORT[$i]}"
    
    if python -c "import $package_import" 2>/dev/null; then
        VERSION=$(python -c "import $package_import; print(getattr($package_import, '__version__', 'unknown'))" 2>/dev/null)
        print_info "$package_install ($VERSION) is installed"
    else
        print_warning "$package_install is not installed"
        MISSING_PACKAGES+=("$package_install")
    fi
done

if [ ${#MISSING_PACKAGES[@]} -ne 0 ]; then
    echo
    print_warning "Some required packages are missing: ${MISSING_PACKAGES[*]}"
    read -p "Do you want to install missing packages now? [y/N]: " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        print_info "Installing missing packages..."
        
        if [ -f "requirements.txt" ]; then
            pip install -r requirements.txt
        else
            pip install torch transformers peft trl datasets accelerate bitsandbytes gradio deepeval wandb pyyaml pandas google-generativeai tenacity unsloth
        fi
        
        print_info "Packages installed successfully"
    else
        print_warning "Skipping package installation"
        print_warning "You can install them later with: pip install -r requirements.txt"
    fi
fi

print_step "Step 6: Checking environment variables..."
if [ -n "$WANDB_API_KEY" ]; then
    print_info "WANDB_API_KEY is set"
else
    print_warning "WANDB_API_KEY is not set"
    print_info "You can add it to .env file or set with: export WANDB_API_KEY='your-key'"
    print_info "Or run training with: ./scripts/run_training.sh --no-wandb"
fi

print_step "Step 7: Verifying configuration files..."
CONFIG_FILES=(
    "configs/sft_agq_9k.yaml"
    "configs/sft_diploy_8B.yaml"
    "configs/research/sft_multi-turn_unsloth_guide.yaml"
    "configs/research/sft_qwen3_single_turn.yaml"
)

for config in "${CONFIG_FILES[@]}"; do
    if [ -f "$config" ]; then
        print_info "Configuration file found: $config"
        if python -c "import yaml; yaml.safe_load(open('$config'))" 2>/dev/null; then
            print_info "  [DONE] Valid YAML syntax"
        else
            print_error "  [FAIL] YAML syntax errors"
        fi
    else
        print_warning "Configuration file not found: $config"
    fi
done


print_step "Step 8: Verifying Python scripts..."
PYTHON_SCRIPTS=(
    "src/training/train_unsloth_multi-turn.py"
    "src/training/train_unsloth_single-turn.py"
    "src/inference/gradio_inference.py"
    "src/eval/deepeval_my_model.py"
)

for script in "${PYTHON_SCRIPTS[@]}"; do
    if [ -f "$script" ]; then
        print_info "Found: $script"
        if python -m py_compile "$script" 2>/dev/null; then
            print_info "  [DONE] No syntax errors"
        else
            print_error "  [FAIL] Syntax errors found"
        fi
    else
        print_warning "Script not found: $script"
    fi
done


print_step "Step 9: Detecting existing models..."
MODEL_COUNT=0
if compgen -G "src/training/sft-*" > /dev/null; then
    print_info "Models found in src/training/:"
    for model in src/training/SFT-*/; do
        if [ -d "$model" ]; then
            print_info "  - $model"
            ((MODEL_COUNT++))
        fi
    done
fi

if compgen -G "src/utils/sft-*" > /dev/null; then
    print_info "Models found in src/utils/:"
    for model in src/utils/SFT-*/; do
        if [ -d "$model" ]; then
            print_info "  - $model"
            ((MODEL_COUNT++))
        fi
    done
fi

if [ $MODEL_COUNT -eq 0 ]; then
    print_warning "No trained models found"
    print_info "Run training to create a model: ./scripts/run_training.sh"
fi


print_step "Step 10: Checking environment template..."

ENV_FILE=".env.template"

if [ -f "$ENV_FILE" ]; then
    print_info "Environment template already exists: $ENV_FILE"
else
    cat > "$ENV_FILE" << 'EOF'
# env var must be set before running any scripts
# cpy this file to .env and fill in the values

WANDB_API_KEY="your-wandb-api-key"
WANDB_ENTITY="your-wandb-entity"
WANDB_PROJECT="qwen3-fine-tuning"
GEMINI_API_KEY="your-gemini-api-key"
HF_TOKEN="your-huggingface-token"
PYTHONPATH="${PYTHONPATH}:$(pwd)"
OPENROUTER_API_KEY="your-openrouter-api-key"
EOF
    print_info "Created environment template: $ENV_FILE"
fi


print_header "Setup Complete!"

echo "Project structure verified"
echo "All shell scripts are executable"
echo "Python installation verified"
echo "Dependencies checked"
echo

if [ -f ".env" ]; then
    print_info "Environment variables loaded from .env file"
else
    print_warning "Create .env file to persist your API keys:"
    echo "  cp .env.template .env"
    echo "  nano .env"
fi

echo
print_info "Next steps:"
echo "  1. If you haven't already, configure your .env file:"
echo "     nano .env"
echo
echo "  2. Review and modify configuration:"
echo "     nano configs/sft_diploy_8B.yaml"
echo
echo "  3. Start the pipeline:"
echo "     ./scripts/run_pipeline.sh"
echo
echo "  Or run individual steps:"
echo "     ./scripts/run_training.sh                           # Train model"
echo "     ./scripts/run_inference.sh -m ./model-path          # Launch Gradio interface"
echo "     ./scripts/run_evaluation.sh -m ./model-path         # Evaluate (OpenAI)"
echo "     ./scripts/run_evaluation_gemini.sh -m ./model-path  # Evaluate (Gemini)"
echo

print_info "For detailed usage, run any script with --help"
echo

if [ -t 0 ]; then
    read -p "Would you like to open the interactive menu now? [y/N]: " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        ./scripts/run_pipeline.sh
    fi
fi