# Configs

## Training Configurations

This directory contains YAML configuration files for training experiments. Each configuration file defines all hyperparameters, model settings, and training arguments for a specific experiment.

##  Directory Structure

```
configs/
├── README.md                      # This file
├── README_QWEN2_7B.md            # Detailed guide for Qwen2-7B
├── sft_qwen2_7B.yaml             # Configuration for Qwen2 7B (RECOMMENDED FOR PRODUCTION)
├── sft_qwen3_1_7B_improved.yaml  # Configuration for Qwen3 1.7B multi-turn
├── sft_llama3.2_1B.yaml          # Configuration for Llama 3.2 1B
```

##  Quick Start

### Using a Configuration

```bash
# Qwen2-7B (recommended for production)
python src/training/training_script_qwen3_improved.py --config configs/sft_qwen2_7B.yaml

# Qwen3 1.7B (faster, development)
python src/training/training_script_qwen3_improved.py --config configs/sft_qwen3_1_7B_improved.yaml

# Or use the general training script
./scripts/run_training.sh
```

### Creating a New Configuration

1. Copy an existing config:
   ```bash
   cp configs/sft_llama32_1b.yaml configs/my_experiment.yaml
   ```

2. Edit the parameters you want to change

3. Run with your new config:
   ```bash
   bash scripts/run_sft.sh my_experiment.yaml
   ```

## 📋 Configuration File Structure

Each YAML config contains the following sections:

### 1. Model Configuration
```yaml
model:
  name: "meta-llama/Llama-3.2-1B"
  trust_remote_code: true
  use_cache: false
```

### 2. Tokenizer Configuration
```yaml
tokenizer:
  padding_side: "right"
  trust_remote_code: true
```

### 3. Quantization Configuration
```yaml
quantization:
  load_in_4bit: true
  bnb_4bit_quant_type: "nf4"
  bnb_4bit_compute_dtype: "float16"
  bnb_4bit_use_double_quant: true
```

### 4. LoRA Configuration
```yaml
lora:
  r: 16                    # Rank (lower = less parameters)
  lora_alpha: 32           # Scaling factor
  target_modules:          # Which layers to adapt
    - "q_proj"
    - "k_proj"
    # ... more modules
  lora_dropout: 0.05
  bias: "none"
  task_type: "CAUSAL_LM"
```

### 5. Dataset Configuration
```yaml
dataset:
  name: "trl-lib/Capybara"
  split: "train"
  test_size: 0.02
  seed: 42
  max_length: 1024         # Maximum sequence length
```

### 6. Training Configuration
```yaml
training:
  output_dir: "./output"
  num_train_epochs: 3
  learning_rate: 0.0002
  per_device_train_batch_size: 1
  gradient_accumulation_steps: 8
  # ... many more parameters
```

### 7. Callbacks Configuration
```yaml
callbacks:
  early_stopping:
    enabled: true
    patience: 3
  memory_monitor:
    enabled: true
    log_every_n_steps: 100
```

### 8. Paths Configuration
```yaml
paths:
  final_model_dir: "./model-output"
```

## 🔧 Common Configuration Scenarios

### Scenario 1: Out of Memory (OOM) Error

**Problem**: GPU runs out of memory during training

**Solution**: Edit these parameters in your config:

```yaml
dataset:
  max_length: 512              # Reduce from 1024

lora:
  r: 8                         # Reduce from 16

training:
  per_device_train_batch_size: 1
  gradient_accumulation_steps: 16  # Increase from 8
```

### Scenario 2: Faster Training (Less Accuracy)

**Goal**: Quick experiments or prototyping

```yaml
dataset:
  test_size: 0.01              # Smaller eval set

training:
  num_train_epochs: 1          # Fewer epochs
  per_device_train_batch_size: 2  # Larger batch (if memory allows)
  eval_steps: 500              # Evaluate less often
  save_steps: 500              # Save less often
  logging_steps: 50            # Log less often
```

### Scenario 3: Higher Quality Model

**Goal**: Best possible results

```yaml
lora:
  r: 32                        # Increase from 16
  lora_alpha: 64               # Increase from 32

training:
  num_train_epochs: 5          # More epochs
  learning_rate: 0.0001        # Lower learning rate
  warmup_ratio: 0.05           # More warmup
  eval_steps: 100              # Evaluate more often
  
callbacks:
  early_stopping:
    patience: 5                # More patience
```

### Scenario 4: Different Dataset

**Goal**: Train on your own dataset

```yaml
dataset:
  name: "your-username/your-dataset"
  split: "train"
  test_size: 0.05
  max_length: 2048             # Adjust based on your data
```

### Scenario 5: Different Model

**Goal**: Use a different base model

```yaml
model:
  name: "meta-llama/Llama-3.2-3B"  # Larger model

# You may need to adjust memory settings:
lora:
  r: 8                         # Reduce for larger model

training:
  gradient_accumulation_steps: 16
```

## 📊 Parameter Reference Guide

### Critical Memory Parameters

| Parameter | Impact | Recommendation |
|-----------|--------|----------------|
| `max_length` |  High | Start at 1024, reduce if OOM |
| `per_device_train_batch_size` |  High | Keep at 1 for 9GB GPU |
| `gradient_accumulation_steps` | 🟡 Medium | Increase to compensate for small batch |
| `lora.r` | 🟡 Medium | 8-16 for 1B models, 4-8 for 3B+ |

### Training Quality Parameters

| Parameter | Effect | Typical Range |
|-----------|--------|---------------|
| `learning_rate` | Higher = faster but less stable | 1e-4 to 5e-4 |
| `num_train_epochs` | More = better fit (risk overfitting) | 1-5 |
| `warmup_ratio` | Stabilizes early training | 0.03-0.1 |
| `lora.r` | Higher = more capacity | 8-64 |
| `lora.alpha` | Scaling factor (usually 2x r) | 16-128 |

### Efficiency Parameters

| Parameter | Effect | Notes |
|-----------|--------|-------|
| `gradient_checkpointing` | Saves memory, slower training | Keep `true` |
| `dataloader_num_workers` | Faster data loading | Set to 0 on small GPU |
| `eval_steps` | How often to evaluate | 100-500 |
| `save_total_limit` | How many checkpoints to keep | 2-3 |

##  Configuration Templates

### Template: Fast Prototyping

```yaml
# Fast iteration for testing
training:
  num_train_epochs: 1
  per_device_train_batch_size: 2
  eval_steps: 500
  save_steps: 500

dataset:
  max_length: 512
  test_size: 0.01

callbacks:
  early_stopping:
    enabled: false
```

### Template: Production Quality

```yaml
# High quality for deployment
training:
  num_train_epochs: 5
  learning_rate: 0.0001
  warmup_ratio: 0.05
  eval_steps: 100
  save_steps: 100

lora:
  r: 32
  lora_alpha: 64

callbacks:
  early_stopping:
    enabled: true
    patience: 5
```

### Template: Memory Constrained

```yaml
# For limited GPU memory
dataset:
  max_length: 512

lora:
  r: 8
  lora_alpha: 16

training:
  per_device_train_batch_size: 1
  gradient_accumulation_steps: 16
```

##  Naming Conventions

Follow these conventions when creating new configs:

```
sft_{model_name}_{variant}.yaml

Examples:
- sft_llama32_1b.yaml              # Base configuration
- sft_llama32_1b_fast.yaml         # Fast training variant
- sft_llama32_1b_high_quality.yaml # High quality variant
- sft_llama32_3b.yaml              # Different model size
- sft_llama32_1b_custom_data.yaml  # Custom dataset
```

##  Version Control Best Practices

### Do's 
- Commit each config file separately
- Use descriptive commit messages
- Document major parameter changes
- Keep a changelog in config comments

### Don'ts 
- Don't commit temporary test configs
- Don't use generic names like `config.yaml`
- Don't delete old configs that produced good results

### Example Commit Message
```
Add high-quality training config for Llama 3.2 1B

- Increased LoRA rank to 32
- Reduced learning rate to 1e-4
- Added 5 epochs with patience=5
- Target: production deployment quality
```

## 🧪 Experiment Tracking

### Organizing Experiments

Create configs for different experiment types:

```
configs/
├── baseline/
│   └── sft_llama32_1b_baseline.yaml
├── learning_rate/
│   ├── sft_llama32_1b_lr_1e4.yaml
│   ├── sft_llama32_1b_lr_2e4.yaml
│   └── sft_llama32_1b_lr_5e4.yaml
└── lora_rank/
    ├── sft_llama32_1b_rank8.yaml
    ├── sft_llama32_1b_rank16.yaml
    └── sft_llama32_1b_rank32.yaml
```

### Config Comparison

To compare configs:

```bash
# View differences
diff configs/sft_llama32_1b.yaml configs/sft_llama32_1b_fast.yaml

# Or use a better diff tool
git diff --no-index configs/sft_llama32_1b.yaml configs/sft_llama32_1b_fast.yaml
```

# Configuration Comparison Guide

Quick reference to help you choose the right configuration for your needs.

## 📊 Available Configurations

| Config | Use Case | Training Time | Memory Usage | Quality |
|--------|----------|---------------|--------------|---------|
| `sft_llama32_1b.yaml` | **Baseline** - Balanced settings | ~3-4 hours | ~9GB | Good |
| `sft_llama32_1b_fast.yaml` | **Quick testing** - Fast iterations | ~1 hour | ~8GB | Fair |
| `sft_llama32_1b_high_quality.yaml` | **Production** - Best results | ~6-8 hours | ~10GB | Excellent |
| `sft_llama32_1b_low_memory.yaml` | **Limited GPU** - 6-8GB GPUs | ~4-5 hours | ~7GB | Good |

## 🔍 Detailed Comparison

### Key Parameters

| Parameter | Baseline | Fast | High Quality | Low Memory |
|-----------|----------|------|--------------|------------|
| **LoRA Rank** | 16 | 8 | 32 | 8 |
| **LoRA Alpha** | 32 | 16 | 64 | 16 |
| **Max Length** | 1024 | 512 | 1024 | 512 |
| **Epochs** | 3 | 1 | 5 | 3 |
| **Learning Rate** | 2e-4 | 3e-4 | 1e-4 | 2e-4 |
| **Batch Size** | 1 | 2 | 1 | 1 |
| **Grad Accum** | 8 | 4 | 16 | 32 |
| **Eval Steps** | 200 | 500 | 100 | 500 |
| **Early Stop Patience** | 3 | Disabled | 5 | 3 |

### Effective Batch Size

Effective batch size = `per_device_train_batch_size × gradient_accumulation_steps`

| Config | Effective Batch Size |
|--------|---------------------|
| Baseline | 1 × 8 = **8** |
| Fast | 2 × 4 = **8** |
| High Quality | 1 × 16 = **16** |
| Low Memory | 1 × 32 = **32** |

##  Decision Tree

```
Start Here
    |
    v
Do you have <8GB GPU memory?
    |
    ├─ YES → Use: sft_llama32_1b_low_memory.yaml
    |
    └─ NO → Continue
        |
        v
    What's your priority?
        |
        ├─ Speed/Testing → Use: sft_llama32_1b_fast.yaml
        |
        ├─ Production/Best Quality → Use: sft_llama32_1b_high_quality.yaml
        |
        └─ Balanced → Use: sft_llama32_1b.yaml (baseline)
```

##  Use Case Examples

### Scenario 1: Initial Experiment
**Goal**: Test if the approach works

**Recommendation**: `sft_llama32_1b_fast.yaml`
- Quick results (~1 hour)
- Low resource usage
- Good for prototyping

### Scenario 2: Hyperparameter Tuning
**Goal**: Find best learning rate

**Recommendation**: `sft_llama32_1b_fast.yaml` (multiple runs)
- Run 5-10 experiments quickly
- Compare results in W&B
- Then use best settings with high quality config

### Scenario 3: Final Model for Deployment
**Goal**: Best possible model

**Recommendation**: `sft_llama32_1b_high_quality.yaml`
- Maximum quality
- More training time
- Careful evaluation

### Scenario 4: Limited Resources
**Goal**: Train on 6GB GPU

**Recommendation**: `sft_llama32_1b_low_memory.yaml`
- Optimized for small GPUs
- May take longer
- Still produces good results

### Scenario 5: Continuous Integration
**Goal**: Automated testing pipeline

**Recommendation**: `sft_llama32_1b_fast.yaml`
- Fast feedback
- Catches major issues
- Good for CI/CD

##  Migration Path

### From Fast → Baseline
```yaml
# In sft_llama32_1b_fast.yaml, change:
lora:
  r: 8 → 16
  lora_alpha: 16 → 32
dataset:
  max_length: 512 → 1024
training:
  num_train_epochs: 1 → 3
  eval_steps: 500 → 200
```

### From Baseline → High Quality
```yaml
# In sft_llama32_1b.yaml, change:
lora:
  r: 16 → 32
  lora_alpha: 32 → 64
training:
  num_train_epochs: 3 → 5
  learning_rate: 0.0002 → 0.0001
  gradient_accumulation_steps: 8 → 16
callbacks:
  early_stopping:
    patience: 3 → 5
```

### From Low Memory → Baseline
```yaml
# In sft_llama32_1b_low_memory.yaml, change:
lora:
  target_modules: [add k_proj, gate_proj, up_proj, down_proj]
dataset:
  max_length: 512 → 1024
training:
  gradient_accumulation_steps: 32 → 8
```

##  Cost Estimation

### Cloud GPU Costs (Approximate)

**Using A10 GPU (~$0.75/hour):**

| Config | Training Time | Estimated Cost |
|--------|---------------|----------------|
| Fast | 1 hour | $0.75 |
| Baseline | 3-4 hours | $2.25-$3.00 |
| High Quality | 6-8 hours | $4.50-$6.00 |
| Low Memory | 4-5 hours | $3.00-$3.75 |

**Using T4 GPU (~$0.35/hour):**

| Config | Training Time | Estimated Cost |
|--------|---------------|----------------|
| Fast | 1.5 hours | $0.53 |
| Baseline | 5-6 hours | $1.75-$2.10 |
| High Quality | 10-12 hours | $3.50-$4.20 |
| Low Memory | 6-7 hours | $2.10-$2.45 |

##  Performance Expectations

### Model Quality (Subjective)

```
Low Memory:    ████████░░  80%
Fast:          ███████░░░  70%
Baseline:      █████████░  90%
High Quality:  ██████████ 100%
```

### Training Speed (Relative)

```
Fast:          ██████████ 100% (fastest)
Low Memory:    ███████░░░  70%
Baseline:      █████░░░░░  50%
High Quality:  ███░░░░░░░  30% (slowest)
```

### Resource Usage

```
Low Memory:    ███████░░░  70%
Fast:          ████████░░  80%
Baseline:      █████████░  90%
High Quality:  ██████████ 100% (most resources)
```

## 🎓 Best Practices

### Development Phase
1. Start with **Fast config**
2. Verify training pipeline works
3. Check data loading and formatting
4. Identify any issues early

### Experimentation Phase
1. Use **Fast config** for multiple runs
2. Test different hyperparameters
3. Track results in W&B
4. Identify best settings

### Production Phase
1. Use **High Quality config**
2. Train with best hyperparameters
3. Monitor closely
4. Validate extensively
5. Save artifacts properly

## 🔧 Customization Tips

### Mix and Match

You can combine settings from different configs:

```yaml
# Example: Fast training with high quality LoRA
# Take from sft_llama32_1b_fast.yaml:
training:
  num_train_epochs: 1
  eval_steps: 500

# Take from sft_llama32_1b_high_quality.yaml:
lora:
  r: 32
  lora_alpha: 64
```

### Create Your Own Variant

```bash
# Copy base config
cp configs/sft_llama32_1b.yaml configs/sft_llama32_1b_custom.yaml

# Edit specific parameters
# Test and iterate
# Document what you changed and why
```

##  Monitoring During Training

### What to Watch

| Config | Key Metrics to Monitor |
|--------|----------------------|
| Fast | Loss decreasing? Pipeline working? |
| Baseline | Eval loss, training stability |
| High Quality | Eval loss, no overfitting, convergence |
| Low Memory | GPU memory usage, OOM errors |

## 🆘 When to Switch Configs

### Switch to Low Memory if:
- Getting OOM errors
- GPU memory >95% used
- Training crashes unexpectedly

### Switch to Fast if:
- Need quick feedback
- Testing major changes
- Running many experiments

### Switch to High Quality if:
- Fast results look promising
- Need production model
- Have time and resources

### Switch to Baseline if:
- Fast is too rough
- High Quality too slow
- Want balanced approach

##  Summary

Choose your config based on:
1. **Available GPU memory**
2. **Time constraints**
3. **Quality requirements**
4. **Stage of development**

Remember: You can always start with Fast, validate with Baseline, and deploy with High Quality!

---

**Quick Reference Command:**

```bash
# Fast iteration
bash scripts/run_sft.sh sft_llama32_1b_fast.yaml

# Balanced training
bash scripts/run_sft.sh sft_llama32_1b.yaml

# Best quality
bash scripts/run_sft.sh sft_llama32_1b_high_quality.yaml

# Limited memory
bash scripts/run_sft.sh sft_llama32_1b_low_memory.yaml
```

##  Troubleshooting

### Config Not Found

```bash
# Check if file exists
ls -la configs/

# Run with explicit path
bash scripts/run_sft.sh sft_llama32_1b.yaml
```

### Invalid YAML Syntax

```bash
# Validate YAML syntax
python3 -c "import yaml; yaml.safe_load(open('configs/sft_llama32_1b.yaml'))"
```

### Parameter Not Applied

1. Check YAML indentation (use spaces, not tabs)
2. Verify parameter name spelling
3. Ensure the parameter is loaded in `sft_train.py`

##  Additional Resources

### Documentation
- [Transformers Training Arguments](https://huggingface.co/docs/transformers/main_classes/trainer#transformers.TrainingArguments)
- [PEFT LoRA Guide](https://huggingface.co/docs/peft/conceptual_guides/lora)
- [TRL SFTTrainer](https://huggingface.co/docs/trl/sft_trainer)

### Related Files
- `src/training/train_script_qwen3_improved.py` - Main training script
- `scripts/run_training.sh` - Bash script to run training
- `requirements.txt` - Required packages

##  Tips & Tricks

1. **Start with base config**: Copy `sft_llama32_1b.yaml` as starting point
2. **Change one thing at a time**: Easier to identify what works
3. **Document your changes**: Add comments in the YAML
4. **Use meaningful names**: `sft_llama32_1b_high_lr.yaml` not `test1.yaml`
5. **Track results**: Use Weights & Biases to compare runs
6. **Keep successful configs**: Don't delete configs that worked well

##  Learning Path

1. **Beginner**: Use base config as-is
2. **Intermediate**: Adjust batch size and learning rate
3. **Advanced**: Experiment with LoRA rank and architecture
4. **Expert**: Create custom configs for specific use cases

##  Support

If you need help with configurations:
1. Check this README
2. Review the SETUP_INSTRUCTIONS.md in project root
3. Examine the training logs for clues
4. Compare with working configs

---

**Happy experimenting!**
