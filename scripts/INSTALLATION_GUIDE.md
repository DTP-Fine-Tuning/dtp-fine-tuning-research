# Quick Installation Guide

## Step 1: Download All Files

Download all the revised shell scripts and documentation from this conversation.

## Step 2: Backup Your Current Scripts

```bash
cd ~/dtp-fine-tuning-research
mkdir -p backup_scripts_$(date +%Y%m%d)
cp scripts/*.sh backup_scripts_$(date +%Y%m%d)/
```

## Step 3: Replace Scripts

```bash
# Assuming you downloaded files to ~/Downloads
cd ~/dtp-fine-tuning-research

# Copy all new scripts to scripts directory
cp ~/Downloads/setup.sh scripts/
cp ~/Downloads/setup_gemini.sh scripts/
cp ~/Downloads/run_training.sh scripts/
cp ~/Downloads/run_inference.sh scripts/
cp ~/Downloads/run_evaluation.sh scripts/
cp ~/Downloads/run_evaluation_gemini.sh scripts/
cp ~/Downloads/run_pipeline.sh scripts/

# Make them executable
chmod +x scripts/*.sh
```

## Step 4: Verify Installation

```bash
cd ~/dtp-fine-tuning-research
./scripts/setup.sh
```

Expected output:
- ✓ Directory exists: configs/
- ✓ Directory exists: scripts/
- ✓ Directory exists: src/
- ✓ Directory exists: src/training/
- ✓ Directory exists: src/utils/
- ✓ Made scripts/run_pipeline.sh executable
- etc.

## Step 5: Test Model Detection

```bash
./scripts/run_inference.sh --help
```

Should show help without errors and display correct paths.

## Step 6: Test Pipeline

```bash
./scripts/run_pipeline.sh
```

Should show interactive menu and list your existing models from both:
- src/training/SFT-*/
- src/utils/SFT-*/

## Troubleshooting

### Issue: "Permission denied"
```bash
chmod +x scripts/*.sh
```

### Issue: "No such file or directory"
Make sure you're in the project root:
```bash
cd ~/dtp-fine-tuning-research
pwd  # Should show: /home/youruser/dtp-fine-tuning-research
```

### Issue: "Python script not found"
Verify your Python scripts are in the correct location:
```bash
ls src/training/*.py
```

Should show:
- src/training/training_script_qwen3_improved.py
- src/training/gradio_inference.py
- src/training/deepeval_evaluation.py
- src/training/deepeval_evaluation_gemini.py

If they're in the wrong location, move them:
```bash
mv training_script_qwen3_improved.py src/training/
mv gradio_inference.py src/training/
mv deepeval_evaluation.py src/training/
mv deepeval_evaluation_gemini.py src/training/
```

## What Changed?

All path references have been updated to match your directory structure:
- Config files: `configs/*.yaml`
- Python scripts: `src/training/*.py`
- Shell scripts: `scripts/*.sh`
- Models: `src/training/SFT-*` and `src/utils/SFT-*`
- Logs: `logs/`
- Results: `evaluation_results/`

## Usage Examples

### Run Training
```bash
./scripts/run_training.sh
```

### Run Inference (auto-detects latest model)
```bash
./scripts/run_inference.sh
```

### Run Evaluation with Gemini (auto-detects latest model)
```bash
./scripts/run_evaluation_gemini.sh -s
```

### Run Complete Pipeline
```bash
./scripts/run_pipeline.sh
```

For detailed information, see SCRIPTS_REVISION_SUMMARY.md
