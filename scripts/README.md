# Shell Scripts Package for Qwen3 Fine-tuning Pipeline

## Package Contents

This package contains all revised shell scripts for your `dtp-fine-tuning-research` project with correct path references and improved functionality.

### Shell Scripts (7 files)
1. **setup.sh** - Main setup and verification script
2. **setup_gemini.sh** - Gemini API configuration
3. **run_training.sh** - Training pipeline script
4. **run_inference.sh** - Gradio inference server
5. **run_evaluation.sh** - OpenAI-based evaluation
6. **run_evaluation_gemini.sh** - Gemini-based evaluation
7. **run_pipeline.sh** - Master orchestration script

### Documentation (3 files)
1. **INSTALLATION_GUIDE.md** - Quick start installation steps
2. **SCRIPTS_REVISION_SUMMARY.md** - Detailed changes and usage guide
3. **DEPLOYMENT_CHECKLIST.md** - Testing and verification checklist

##  What Was Fixed

### 1. **Path Corrections**
- Config files now correctly reference `configs/*.yaml`
- Python scripts now correctly reference `src/training/*.py`
- Shell scripts now correctly reference `scripts/*.sh`
- Models searched in both `src/training/SFT-*` and `src/utils/SFT-*`

### 2. **Project Root Verification**
- All scripts verify they're run from project root
- Clear error messages if run from wrong directory
- Helpful guidance on correct usage

### 3. **Smart Model Detection**
-  Auto-detects latest model from both training and utils directories
-  Lists all available models for user selection
-  Handles missing models gracefully

### 4. **Directory Management**
- setup.sh now verifies instead of recreating existing directories
- Only creates optional directories (logs, evaluation_results)
- Reports status accurately

### 5. **Better Error Handling**
- Clear, actionable error messages
- Suggests alternatives when files not found
- Shows current state and expected state

## Quick Start

### Step 1: Backup Current Scripts
```bash
cd ~/dtp-fine-tuning-research
mkdir -p backup_scripts_$(date +%Y%m%d)
cp scripts/*.sh backup_scripts_$(date +%Y%m%d)/
```

### Step 2: Install New Scripts
```bash
# Copy downloaded scripts to your scripts directory
cp /path/to/downloaded/*.sh scripts/

# Make executable
chmod +x scripts/*.sh
```

### Step 3: Verify Installation
```bash
./scripts/setup.sh
```

### Step 4: Test
```bash
./scripts/run_pipeline.sh
```

## File Descriptions

### setup.sh
- Verifies project structure
- Checks Python dependencies
- Validates environment variables
- Makes all scripts executable
- **Key Change:** Now verifies instead of creating existing directories

### setup_gemini.sh
- Configures Gemini API
- Tests API connection
- Saves API key to environment
- **Key Change:** Updated paths to `scripts/` and `src/training/`

### run_training.sh
- Runs model training
- Validates config files
- Creates logs
- **Key Changes:** 
  - Default config: `configs/sft_qwen3_1_7B_improved.yaml`
  - Default script: `src/training/training_script_qwen3_improved.py`

### run_inference.sh
- Launches Gradio web interface
- Auto-detects latest model
- Displays model information
- **Key Changes:**
  - Default script: `src/training/gradio_inference.py`
  - Searches models in both `src/training/` and `src/utils/`

### run_evaluation.sh
- Evaluates model with OpenAI metrics
- Supports sample data and custom datasets
- Generates reports
- **Key Changes:**
  - Default script: `src/training/deepeval_evaluation.py`
  - Auto-detects models from both locations

### run_evaluation_gemini.sh
- Evaluates model with Gemini metrics
- Supports multiple Gemini models
- Generates reports
- **Key Changes:**
  - Default script: `src/training/deepeval_evaluation_gemini.py`
  - Logs to `evaluation_gemini_*.log`

### run_pipeline.sh
- Interactive menu system
- Orchestrates all operations
- Manages model selection
- **Key Changes:**
  - All script references now include `scripts/` prefix
  - Model detection from both locations
  - Better error handling

## Detailed Documentation

### For Installation
See **INSTALLATION_GUIDE.md**

### For Complete Changes
See **SCRIPTS_REVISION_SUMMARY.md** (16KB, comprehensive)

### For Testing
See **DEPLOYMENT_CHECKLIST.md**

## Usage Examples

### Interactive Mode
```bash
./scripts/run_pipeline.sh
# Shows menu, auto-detects models
```

### Direct Commands
```bash
# Training
./scripts/run_training.sh

# Inference (auto-detect model)
./scripts/run_inference.sh

# Inference (specific model)
./scripts/run_inference.sh -m src/training/SFT-Qwen3-1.7B-LoRA-9GB-final

# Evaluation with Gemini (quick test)
./scripts/run_evaluation_gemini.sh -s

# Evaluation with OpenAI (dataset)
./scripts/run_evaluation.sh -d "izzulgod/indonesian-conversation" -n 100
```

### Pipeline Operations
```bash
# Complete pipeline
./scripts/run_pipeline.sh pipeline

# Individual steps
./scripts/run_pipeline.sh train
./scripts/run_pipeline.sh inference
./scripts/run_pipeline.sh evaluate
./scripts/run_pipeline.sh evaluate-gemini
./scripts/run_pipeline.sh quick-test-gemini
```

## Requirements

### Directory Structure
```
dtp-fine-tuning-research/
├── configs/                  # Config YAML files
├── scripts/                  # Shell scripts (NEW LOCATION)
├── src/
│   ├── training/            # Python scripts + models
│   └── utils/               # Utility models
├── logs/                    # Auto-created if missing
└── evaluation_results/      # Auto-created if missing
```

### Python Scripts Expected Location
- `src/training/training_script_qwen3_improved.py`
- `src/training/gradio_inference.py`
- `src/training/deepeval_evaluation.py`
- `src/training/deepeval_evaluation_gemini.py`

### Environment Variables (Optional)
```bash
export WANDB_API_KEY="your-key"
export OPENAI_API_KEY="your-key"  # For OpenAI evaluation
export GEMINI_API_KEY="your-key"  # For Gemini evaluation
```

## Troubleshooting

### Scripts Won't Run
```bash
chmod +x scripts/*.sh
```

### "Must be run from project root"
```bash
cd ~/dtp-fine-tuning-research
# Always run from here
```

### "No models found"
```bash
# Check model locations
ls src/training/SFT-*/
ls src/utils/SFT-*/
```

### "Config/Script not found"
```bash
# Verify files are in correct locations
ls configs/
ls src/training/*.py
```

## Migration from Old Scripts

1. Your Python scripts should be in `src/training/` directory
2. Your config files should be in `configs/` directory
3. Your shell scripts should be in `scripts/` directory
4. Your models can be in either `src/training/SFT-*` or `src/utils/SFT-*`

If files are in wrong locations, move them:
```bash
# Example: Move Python scripts
mv *.py src/training/

# Example: Move configs
mv *.yaml configs/

# Example: Move models
mv SFT-* src/training/
```

## Verification

After installation, verify everything works:

```bash
cd ~/dtp-fine-tuning-research

# 1. Setup should succeed
./scripts/setup.sh

# 2. Help should show without errors
./scripts/run_training.sh --help
./scripts/run_inference.sh --help
./scripts/run_evaluation.sh --help
./scripts/run_evaluation_gemini.sh --help

# 3. Pipeline should show menu
./scripts/run_pipeline.sh
```

## Support

If you encounter issues:

1. Check **SCRIPTS_REVISION_SUMMARY.md** for detailed explanations
2. Follow **DEPLOYMENT_CHECKLIST.md** for systematic testing
3. Verify your directory structure matches the expected layout
4. Ensure all Python scripts are in `src/training/`
5. Ensure all config files are in `configs/`

## Acknowledgments

Revised to properly handle the established directory structure of the dtp-fine-tuning-research project, with improved path handling, better error messages, and smart model detection.

---

**Version:** 2.0  
**Date:** November 2024
