#!/bin/bash

################################################################################
# Setup Script for Qwen3 Fine-tuning Pipeline
# This script prepares the environment and makes all scripts executable
################################################################################

set -e

# Color codes
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
    echo -e "${GREEN}✓${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

print_step() {
    echo
    echo -e "${BLUE}➜${NC} $1"
}

################################################################################
# Main Setup
################################################################################

print_header "Qwen3 Fine-tuning Pipeline Setup"

# Check if we're in the right directory
if [ ! -f "run_pipeline.sh" ]; then
    print_error "Setup script must be run from the project root directory"
    print_info "Current directory: $(pwd)"
    exit 1
fi

################################################################################
# 1. Make Scripts Executable
################################################################################

print_step "Step 1: Making scripts executable..."

scripts=(
    "run_pipeline.sh"
    "run_training.sh"
    "run_inference.sh"
    "run_evaluation.sh"
    "run_evaluation_gemini.sh" 
    "setup_gemini.sh"          
)

for script in "${scripts[@]}"; do
    if [ -f "$script" ]; then
        chmod +x "$script"
        print_info "Made $script executable"
    else
        print_warning "$script not found"
    fi
done

################################################################################
# 2. Create Directory Structure
################################################################################

print_step "Step 2: Creating directory structure..."

directories=(
    "logs"
    "evaluation_results"
    "configs"
    "checkpoints"
)

for dir in "${directories[@]}"; do
    if [ ! -d "$dir" ]; then
        mkdir -p "$dir"
        print_info "Created directory: $dir"
    else
        print_info "Directory exists: $dir"
    fi
done

################################################################################
# 3. Check Python Installation
################################################################################

print_step "Step 3: Checking Python installation..."

if ! command -v python &> /dev/null; then
    print_error "Python not found. Please install Python 3.8 or higher."
    exit 1
fi

PYTHON_VERSION=$(python --version 2>&1 | cut -d' ' -f2)
print_info "Python version: $PYTHON_VERSION"

# Check Python version
REQUIRED_VERSION="3.8.0"
if ! python -c "import sys; exit(0 if sys.version_info >= tuple(map(int, '$REQUIRED_VERSION'.split('.'))) else 1)" 2>/dev/null; then
    print_error "Python 3.8 or higher is required"
    exit 1
fi

################################################################################
# 4. Check GPU/CUDA
################################################################################

print_step "Step 4: Checking GPU/CUDA availability..."

if command -v nvidia-smi &> /dev/null; then
    print_info "NVIDIA GPU detected"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | while read line; do
        print_info "  $line"
    done
    
    # Check CUDA availability in Python
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

################################################################################
# 5. Check Python Dependencies
################################################################################

print_step "Step 5: Checking Python dependencies..."

# Daftar nama paket untuk di-install
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
)

MISSING_PACKAGES=()
MISSING_PACKAGES_INSTALL=()

for i in "${!REQUIRED_PACKAGES_INSTALL[@]}"; do
    package_install="${REQUIRED_PACKAGES_INSTALL[$i]}"
    package_import="${REQUIRED_PACKAGES_IMPORT[$i]}"
    
    if python -c "import $package_import" 2>/dev/null; then
        # Get version
        VERSION=$(python -c "import $package_import; print(getattr($package_import, '__version__', 'unknown'))" 2>/dev/null)
        print_info "$package_install ($VERSION) is installed"
    else
        print_warning "$package_install is not installed"
        MISSING_PACKAGES+=("$package_install")
        MISSING_PACKAGES_INSTALL+=("$package_install")
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
            # Install semua paket yang diperlukan, termasuk yang baru
            pip install torch transformers peft trl datasets accelerate bitsandbytes gradio deepeval wandb pyyaml pandas google-generativeai tenacity
        fi
        
        print_info "Packages installed successfully"
    else
        print_warning "Skipping package installation"
        print_warning "You can install them later with: pip install -r requirements.txt"
    fi
fi

################################################################################
# 6. Check Environment Variables
################################################################################

print_step "Step 6: Checking environment variables..."

# Check W&B
if [ -n "$WANDB_API_KEY" ]; then
    print_info "WANDB_API_KEY is set"
else
    print_warning "WANDB_API_KEY is not set"
    print_info "You can set it with: export WANDB_API_KEY='your-key'"
    print_info "Or run training with: ./run_training.sh --no-wandb"
fi

# Check OpenAI
if [ -n "$OPENAI_API_KEY" ]; then
    print_info "OPENAI_API_KEY is set"
else
    print_warning "OPENAI_API_KEY is not set"
    print_info "Required for DeepEval (OpenAI) evaluation metrics"
    print_info "You can set it with: export OPENAI_API_KEY='your-key'"
fi

# Check Gemini
if [ -n "$GEMINI_API_KEY" ] || [ -n "$GOOGLE_API_KEY" ]; then
    print_info "GEMINI_API_KEY or GOOGLE_API_KEY is set"
else
    print_warning "GEMINI_API_KEY / GOOGLE_API_KEY is not set"
    print_info "Required for DeepEval (Gemini) evaluation metrics"
    print_info "Get one at: https://makersuite.google.com/app/apikey"
    print_info "You can set it with: export GEMINI_API_KEY='your-key'"
fi

################################################################################
# 7. Verify Configuration Files
################################################################################

print_step "Step 7: Verifying configuration files..."

CONFIG_FILE="configs/sft_qwen3_1_7B_improved.yaml"

if [ -f "$CONFIG_FILE" ]; then
    print_info "Configuration file found: $CONFIG_FILE"
    
    # Validate YAML
    if python -c "import yaml; yaml.safe_load(open('$CONFIG_FILE'))" 2>/dev/null; then
        print_info "Configuration file is valid YAML"
    else
        print_error "Configuration file has YAML syntax errors"
    fi
else
    print_warning "Configuration file not found: $CONFIG_FILE"
    print_info "Make sure to create it before training"
fi

################################################################################
# 8. Verify Python Scripts
################################################################################

print_step "Step 8: Verifying Python scripts..."

SCRIPTS=(
    "training_script_qwen3_improved.py"
    "gradio_inference.py"
    "deepeval_evaluation.py"
    "deepeval_evaluation_gemini.py"
)

for script in "${SCRIPTS[@]}"; do
    if [ -f "$script" ]; then
        print_info "Found: $script"
        
        # Check for syntax errors
        if python -m py_compile "$script" 2>/dev/null; then
            print_info "  ✓ No syntax errors"
        else
            print_error "  ✗ Syntax errors found"
        fi
    else
        print_warning "Script not found: $script"
    fi
done

################################################################################
# 9. Create Environment File Template
################################################################################

print_step "Step 9: Creating environment template..."

ENV_FILE=".env.template"

cat > "$ENV_FILE" << 'EOF'
# Environment Variables for Qwen3 Fine-tuning Pipeline
# Copy this to .env and fill in your values
# Then run: source .env

# Weights & Biases
export WANDB_API_KEY="your-wandb-api-key"
export WANDB_ENTITY="your-wandb-entity"
export WANDB_PROJECT="qwen3-fine-tuning"

# OpenAI (for DeepEval)
export OPENAI_API_KEY="your-openai-api-key"

# Gemini (for DeepEval) <-- DITAMBAHKAN
export GEMINI_API_KEY="your-gemini-api-key"

# CUDA Configuration (optional)
export CUDA_VISIBLE_DEVICES="0"  # Use specific GPU

# Hugging Face (optional, for private models/datasets)
export HF_TOKEN="your-huggingface-token"

# Python Configuration
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
EOF

print_info "Created environment template: $ENV_FILE"
print_info "Copy to .env and fill in your values"

################################################################################
# 10. Summary
################################################################################

print_header "Setup Complete!"

echo "All shell scripts are executable"
echo "Directory structure created"
echo "Python installation verified"
echo "Dependencies checked"
echo

print_info "Next steps:"
echo "  1. Configure your environment variables (optional):"
echo "     cp .env.template .env"
echo "     nano .env"
echo "     source .env"
echo
echo "  2. Review and modify configuration:"
echo "     nano configs/sft_qwen3_1_7B_improved.yaml"
echo
echo "  3. Start the pipeline:"
echo "     ./run_pipeline.sh"
echo
echo "  Or run individual steps:"
echo "     ./run_training.sh                         # Train model"
echo "     ./run_inference.sh -m ./model-path      # Launch Gradio interface"
echo "     ./run_evaluation.sh -m ./model-path     # Evaluate model (OpenAI)"
echo "     ./run_evaluation_gemini.sh -m ./model-path # Evaluate model (Gemini)"
echo

print_info "For detailed usage instructions, see: SCRIPTS_README.md"
echo

# Check if running interactively
if [ -t 0 ]; then
    read -p "Would you like to open the interactive menu now? [y/N]: " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        ./run_pipeline.sh
    fi
fi