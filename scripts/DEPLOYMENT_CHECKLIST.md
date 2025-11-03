# Deployment Checklist

## Pre-Deployment

- [ ] Backup current scripts
  ```bash
  mkdir -p backup_scripts_$(date +%Y%m%d)
  cp scripts/*.sh backup_scripts_$(date +%Y%m%d)/
  ```

- [ ] Verify project structure matches expected layout
  ```bash
  cd ~/dtp-fine-tuning-research
  ls -la  # Should see: configs/, scripts/, src/, logs/, evaluation_results/
  ```

- [ ] Verify Python scripts are in src/training/
  ```bash
  ls src/training/*.py
  ```

## Deployment

- [ ] Copy new scripts to scripts/ directory
- [ ] Make scripts executable: `chmod +x scripts/*.sh`
- [ ] Run setup: `./scripts/setup.sh`

## Post-Deployment Testing

### Basic Tests

- [ ] Test project root verification
  ```bash
  cd ~/dtp-fine-tuning-research
  ./scripts/setup.sh
  # Should succeed
  
  cd ~
  ./scripts/setup.sh  
  # Should fail with helpful message
  ```

- [ ] Test help commands (all should work without errors)
  ```bash
  cd ~/dtp-fine-tuning-research
  ./scripts/run_training.sh --help
  ./scripts/run_inference.sh --help
  ./scripts/run_evaluation.sh --help
  ./scripts/run_evaluation_gemini.sh --help
  ./scripts/run_pipeline.sh --help
  ```

### Model Detection Tests

- [ ] Test model auto-detection
  ```bash
  ./scripts/run_inference.sh
  # Should list models from both src/training/ and src/utils/
  # Press Ctrl+C to exit
  ```

- [ ] Verify models are found
  ```bash
  # Should show your models:
  # - src/training/SFT-Llama-3.2-1B-LoRA-9GB/
  # - src/training/SFT-Llama-3.2-1B-LoRA-9GB-final/
  # - src/utils/SFT-Qwen3-1.7B-LoRA-9GB/
  # - src/utils/SFT-Qwen3-1.7B-LoRA-9GB-final/
  ```

### Path Validation Tests

- [ ] Test config file detection
  ```bash
  ls configs/
  # Should show your config files
  ```

- [ ] Test Python script detection
  ```bash
  ls src/training/*.py
  # Should show: training_script_qwen3_improved.py, gradio_inference.py, etc.
  ```

- [ ] Test log directory
  ```bash
  ls logs/
  # Should exist
  ```

- [ ] Test evaluation results directory
  ```bash
  ls evaluation_results/
  # Should exist
  ```

## Functional Tests

### If You Have Trained Models

- [ ] Test quick evaluation (Gemini)
  ```bash
  ./scripts/run_evaluation_gemini.sh -s
  # Should run evaluation with sample data
  ```

- [ ] Test quick evaluation (OpenAI)
  ```bash
  ./scripts/run_evaluation.sh -s
  # Should run evaluation with sample data (requires OPENAI_API_KEY)
  ```

- [ ] Test inference
  ```bash
  ./scripts/run_inference.sh
  # Should launch Gradio interface
  # Open browser to http://localhost:7860
  # Press Ctrl+C to stop
  ```

### Pipeline Test

- [ ] Test pipeline menu
  ```bash
  ./scripts/run_pipeline.sh
  # Should show interactive menu
  # Test navigation through menu options
  # Select option 8 to exit
  ```

## Environment Variables

- [ ] Verify API keys are set (if using evaluations)
  ```bash
  echo $WANDB_API_KEY
  echo $OPENAI_API_KEY
  echo $GEMINI_API_KEY
  ```

- [ ] If not set, configure them
  ```bash
  cp .env.template .env
  nano .env  # Edit with your keys
  source .env
  ```

## Common Issues Resolution

### Issue: Scripts show "command not found"
- [ ] Solution: Verify scripts are executable
  ```bash
  chmod +x scripts/*.sh
  ```

### Issue: "Must be run from project root"
- [ ] Solution: Always run from project root
  ```bash
  cd ~/dtp-fine-tuning-research
  # Then run scripts
  ```

### Issue: "Config file not found"
- [ ] Solution: Check config location
  ```bash
  ls configs/
  # Ensure configs are in configs/ directory
  ```

### Issue: "Python script not found"
- [ ] Solution: Check Python script location
  ```bash
  ls src/training/*.py
  # Ensure Python scripts are in src/training/ directory
  ```

### Issue: "No models found"
- [ ] Solution: Check model locations
  ```bash
  ls src/training/SFT-*/
  ls src/utils/SFT-*/
  # Ensure models are in one of these locations
  ```

## Rollback Plan

If something goes wrong:

```bash
cd ~/dtp-fine-tuning-research

# Restore from backup
cp backup_scripts_YYYYMMDD/*.sh scripts/
chmod +x scripts/*.sh

# Verify restoration
./scripts/setup.sh
```

## Final Verification

- [ ] All scripts run without path errors
- [ ] Model detection works correctly
- [ ] Pipeline menu shows all models
- [ ] Help commands display correctly
- [ ] Error messages are clear and actionable

## Sign-off

Date: _______________
Tested by: _______________
Status: [ ] PASS [ ] FAIL
Notes: _______________________________________________
