# SFT Training with YAML Configuration - Setup Instructions

This refactored training pipeline uses YAML configuration files to manage all training parameters.

## Quick Setup

### 1. Place Files in Correct Directories

```bash
# From project root (~/dtp-fine-tuning-research)

# Copy config file to configs directory
cp sft_llama32_1b.yaml configs/

# Copy training script to src/training directory
cp sft_train.py src/training/

# Copy bash script to scripts directory
cp run_sft.sh scripts/
chmod +x scripts/run_sft.sh
```

### 2. Install Dependencies

```bash
pip install pyyaml --break-system-packages
# Other dependencies should already be installed from requirements.txt
```

### 3. Run Training

```bash
# From project root
bash scripts/run_sft.sh

# Or specify a config
bash scripts/run_sft.sh sft_llama32_1b.yaml
```

## What Changed

### Before (Hardcoded):
- All parameters in training_script_llama32_simple.py
- Hard to track experiments
- Need to modify code for changes

### After (YAML-based):
- All parameters in configs/sft_llama32_1b.yaml
- Easy to version control configurations
- No code changes needed for experiments

## File Descriptions

1. **sft_llama32_1b.yaml** - Configuration file with all training parameters
2. **sft_train.py** - Refactored training script that loads config
3. **run_sft.sh** - Bash script to run training with proper path handling
4. **SETUP_INSTRUCTIONS.md** - This file

## Usage Examples

```bash
# Basic usage
bash scripts/run_sft.sh

# With custom config
bash scripts/run_sft.sh my_experiment.yaml

# Direct Python call
python3 src/training/sft_train.py --config configs/sft_llama32_1b.yaml
```

## Benefits

✅ All configs in one YAML file
✅ Easy to create multiple experiment configs
✅ Version control friendly
✅ No code modifications needed
✅ Self-documenting parameters
✅ Bash script handles paths automatically

## Troubleshooting

**Config not found**: Make sure you copied sft_llama32_1b.yaml to configs/ directory
**Script not executable**: Run `chmod +x scripts/run_sft.sh`
**Import errors**: Install pyyaml with `pip install pyyaml --break-system-packages`
