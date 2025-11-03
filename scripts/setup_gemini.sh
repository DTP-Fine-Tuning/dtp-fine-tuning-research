#!/bin/bash

################################################################################
# Setup Script for Gemini Migration
# This script helps set up the environment for using Gemini instead of OpenAI
# Must be run from project root: ~/dtp-fine-tuning-research/
################################################################################

set -e

# Color codes
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${BLUE}╔════════════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║       Gemini API Setup for DeepEval Evaluation Pipeline               ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════════════════╝${NC}"
echo

# Function to print colored messages
print_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

verify_project_root() {
    if [ ! -d "scripts" ] || [ ! -d "src" ] || [ ! -d "configs" ]; then
        print_error "This script must be run from the project root directory"
        print_info "Current directory: $(pwd)"
        print_info "Please cd to project root and run: ./scripts/setup_gemini.sh"
        exit 1
    fi
}

# Verify we're in project root
verify_project_root

# Check Python version
print_info "Checking Python version..."
python_version=$(python --version 2>&1 | grep -oE '[0-9]+\.[0-9]+')
required_version="3.8"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" = "$required_version" ]; then
    print_info "Python version $python_version is compatible ✓"
else
    print_error "Python version $python_version is too old. Please use Python 3.8 or higher."
    exit 1
fi

# Install dependencies
echo
print_info "Installing required Python packages..."
pip install google-generativeai tenacity deepeval --upgrade

if [ $? -eq 0 ]; then
    print_info "Dependencies installed successfully ✓"
else
    print_error "Failed to install dependencies"
    exit 1
fi

# Check for existing API keys
echo
print_info "Checking for existing API keys..."

if [ ! -z "$GEMINI_API_KEY" ]; then
    print_info "Found GEMINI_API_KEY in environment ✓"
    existing_key="$GEMINI_API_KEY"
elif [ ! -z "$GOOGLE_API_KEY" ]; then
    print_info "Found GOOGLE_API_KEY in environment ✓"
    existing_key="$GOOGLE_API_KEY"
else
    print_warning "No Gemini API key found in environment"
    existing_key=""
fi

# API Key setup
echo
echo -e "${BLUE}═══════════════════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}                        API Key Configuration                          ${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════════════════════${NC}"
echo

if [ -z "$existing_key" ]; then
    echo "To get a Gemini API key:"
    echo "1. Visit: https://makersuite.google.com/app/apikey"
    echo "2. Sign in with your Google account"
    echo "3. Click 'Create API Key'"
    echo "4. Copy the generated key"
    echo
    
    read -p "Do you have a Gemini API key ready? (y/n): " has_key
    
    if [[ "$has_key" =~ ^[Yy]$ ]]; then
        read -p "Enter your Gemini API key: " api_key
        
        if [ ! -z "$api_key" ]; then
            # Test the API key
            print_info "Testing API key..."
            
            python -c "
import google.generativeai as genai
try:
    genai.configure(api_key='$api_key')
    model = genai.GenerativeModel('gemini-1.5-flash')
    response = model.generate_content('Hello')
    if response.text:
        print('✓ API key is valid!')
        exit(0)
except Exception as e:
    print(f'✗ API key validation failed: {e}')
    exit(1)
" 
            
            if [ $? -eq 0 ]; then
                print_info "API key validated successfully! ✓"
                
                # Ask if user wants to save to .env or bashrc
                echo
                echo "Where would you like to save the API key?"
                echo "1. Environment file (.env) - Recommended for project"
                echo "2. Shell profile (~/.bashrc) - System-wide"
                echo "3. Don't save, I'll set it manually"
                read -p "Choose option (1/2/3): " save_option
                
                case $save_option in
                    1)
                        echo "GEMINI_API_KEY='$api_key'" >> .env
                        print_info "API key saved to .env file ✓"
                        echo "Load it with: source .env"
                        ;;
                    2)
                        echo "export GEMINI_API_KEY='$api_key'" >> ~/.bashrc
                        print_info "API key saved to ~/.bashrc ✓"
                        echo "Load it with: source ~/.bashrc"
                        ;;
                    3)
                        echo
                        echo "To set the API key manually, run:"
                        echo "export GEMINI_API_KEY='$api_key'"
                        ;;
                esac
                
                export GEMINI_API_KEY="$api_key"
            else
                print_error "API key validation failed. Please check your key and try again."
                exit 1
            fi
        fi
    else
        echo
        print_warning "Please get an API key from https://makersuite.google.com/app/apikey"
        print_warning "Then set it with: export GEMINI_API_KEY='your-key-here'"
    fi
else
    print_info "Using existing API key from environment"
fi

# Make scripts executable
echo
print_info "Making scripts executable..."
if [ -f "scripts/run_evaluation_gemini.sh" ]; then
    chmod +x scripts/run_evaluation_gemini.sh
    print_info "scripts/run_evaluation_gemini.sh is now executable ✓"
else
    print_warning "scripts/run_evaluation_gemini.sh not found"
fi

# Verify Python script exists
echo
print_info "Verifying Gemini evaluation script..."
if [ -f "src/training/deepeval_evaluation_gemini.py" ]; then
    print_info "Found: src/training/deepeval_evaluation_gemini.py ✓"
else
    print_warning "src/training/deepeval_evaluation_gemini.py not found"
    print_warning "Please ensure the Gemini evaluation script is in src/training/"
fi

# Create backup of original files
echo
read -p "Do you want to backup original OpenAI evaluation files? (y/n): " backup_choice

if [[ "$backup_choice" =~ ^[Yy]$ ]]; then
    backup_dir="backup_openai_$(date +%Y%m%d_%H%M%S)"
    mkdir -p "$backup_dir"
    
    if [ -f "src/training/deepeval_evaluation.py" ]; then
        cp src/training/deepeval_evaluation.py "$backup_dir/"
        print_info "Backed up src/training/deepeval_evaluation.py"
    fi
    
    if [ -f "scripts/run_evaluation.sh" ]; then
        cp scripts/run_evaluation.sh "$backup_dir/"
        print_info "Backed up scripts/run_evaluation.sh"
    fi
    
    print_info "Original files backed up to $backup_dir/ ✓"
fi

# Final setup summary
echo
echo -e "${BLUE}═══════════════════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}                         Setup Complete!                               ${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════════════════════${NC}"
echo

print_info "✅ Gemini evaluation environment is ready!"
echo
echo "Next steps:"
echo "1. Test with sample data:"
echo "   ./scripts/run_evaluation_gemini.sh -s"
echo
echo "2. Run full evaluation on a model:"
echo "   ./scripts/run_evaluation_gemini.sh -m src/training/SFT-Qwen3-1.7B-LoRA-9GB-final -s"
echo
echo "3. Run evaluation on a dataset:"
echo "   ./scripts/run_evaluation_gemini.sh -d dataset-name -n 100"
echo
echo "Available Gemini models:"
echo "  - gemini-1.5-flash (recommended - fast & cheap)"
echo "  - gemini-1.5-pro (higher quality)"
echo "  - gemini-1.0-pro (legacy)"
echo
echo "For help: ./scripts/run_evaluation_gemini.sh --help"
echo

# Quick test option
read -p "Would you like to run a quick API test now? (y/n): " test_choice

if [[ "$test_choice" =~ ^[Yy]$ ]]; then
    print_info "Running quick Gemini API test..."
    
    python -c "
import google.generativeai as genai
import os

api_key = os.environ.get('GEMINI_API_KEY') or os.environ.get('GOOGLE_API_KEY')
if not api_key:
    print('Error: No API key found')
    exit(1)

genai.configure(api_key=api_key)
model = genai.GenerativeModel('gemini-1.5-flash')

print('Testing Gemini API...')
response = model.generate_content('Translate to Indonesian: Hello, how are you?')
print(f'Response: {response.text}')
print('✅ Gemini API is working!')
"
fi

echo
print_info "Setup script completed!"
