# Configs

## Training Configurations

This directory contains YAML configuration files for training experiments. Each configuration file defines all hyperparameters, model settings, and training arguments for a specific experiment.

##  Directory Structure

```
configs/
├── research/                      # dirs accomodate research yml file
├── README.md                      # This file
├── sft_diploy_8B.yaml             # Base configuration for multi-turn
├── sft_agq_9k.yaml  # Base configuration for single-turn
```

##  Quick Start

### Using a Configuration

```bash
# From project root for multi-turn
bash scripts/run_sft.sh -c sft_diploy_8B.yaml
```

### Creating a New Configuration

1. Copy an existing config:
   ```bash
   cp configs/sft_diploy_8B.yaml configs/research/my_research_experiments.yaml
   ```

2. Edit the parameters you want to change

3. Run with your new config:
   ```bash
   bash scripts/run_sft.sh -c configs/research/my_research_experiments.yaml
   ```

## Configuration File Structure

Each YAML config contains the following sections:

### 1. Model Configuration
```yaml
model:
  name: "aitfindonesia/KomdigiUB-8B-Base"
  trust_remote_code: true
  use_cache: false
```

### 2. Tokenizer Configuration
```yaml
tokenizer:
  padding_side: "left"
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
  max_length: 4096         # you can edit this max seq len
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

### 8. Paths Configuration
```yaml
paths:
  final_model_dir: "./model-output"
```

## Common Configuration Scenarios

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

## Parameter Reference Guide

### Critical Memory Parameters

| Parameter | Impact | Recommendation |
|-----------|--------|----------------|
| `max_length` |  High | Start at 1024, reduce if OOM |
| `per_device_train_batch_size` |  High | Keep at 1 for 9GB GPU |
| `gradient_accumulation_steps` | Medium | Increase to compensate for small batch |
| `lora.r` | Medium | 8-16 for 1B models, 4-8 for 3B+ |

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

## Experiment Tracking

### Organizing Experiments

Create configs for different experiment types like this example:

```
configs/research/
├── baseline-diploy/
│   └── sft_diploy_8B.yaml
├── learning_rate-diploy/
│   ├── sft_diploy_8B.yaml_lr_1e4.yaml
│   ├── sft_diploy_8B.yaml_lr_2e4.yaml
│   └── sft_diploy_8B.yaml_lr_5e4.yaml
└── lora_rank-diploy/
    ├── sft_diploy_8B.yamlrank8.yaml
    ├── sft_diploy_8B.yamlrank16.yaml
    └── sft_diploy_8B.yamlrank32.yaml
```

### Config Comparison

To compare configs:

```bash
# View differences
diff configs/baseline-diploy/sft_diploy_8B.yaml configs/learning_rate-diploy/sft_diploy_8B.yaml_lr_5e4.yaml

# Or use a better diff tool
git diff --no-index configs/baseline-diploy/sft_diploy_8B.yaml configs/learning_rate-diploy/sft_diploy_8B.yaml_lr_5e4.yaml
```

##  Monitoring During Training

### What to Watch

You can see the documentation of loss_curve in [`loss_curve_guide.md`](docs/loss_curve_guide.md)

| Config | Key Metrics to Monitor |
|--------|----------------------|
| Fast | Loss decreasing? Pipeline working? |
| Baseline | Eval loss, training stability |
| High Quality | Eval loss, no overfitting, convergence |
| Low Memory | GPU memory usage, OOM errors |

## When to Switch Configs

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
# multi-turn training baseline
bash scripts/run_sft.sh -c configs/sft_diploy_8B.yaml

# single-turn training baseline
bash scripts/run_sft.sh -c sft_agq_9k.yaml
```

##  Troubleshooting

### Config Not Found

```bash
# Check if file exists
ls -la configs/

# Run with explicit path
bash scripts/run_sft.sh -c configs/sft_diploy_8B.yaml
```

### Invalid YAML Syntax

```bash
# Validate YAML syntax
python3 -c "import yaml; yaml.safe_load(open('configs/sft_llama32_1b.yaml'))"
```

### Parameter Not Applied

1. Check YAML indentation (use spaces, not tabs)
2. Verify parameter name spelling

##  Additional Resources

### Documentation
- [Transformers Training Arguments](https://huggingface.co/docs/transformers/main_classes/trainer#transformers.TrainingArguments)
- [PEFT LoRA Guide](https://huggingface.co/docs/peft/conceptual_guides/lora)
- [TRL SFTTrainer](https://huggingface.co/docs/trl/sft_trainer)
- [Unsloth](https://unsloth.ai/docs)

### Related Files
- `src/training/train_unsloth_multi-turn.py` - Main training script for multi-turn
- `src/training/train_unsloth_single_turn.py` - Main training script for multi-turn
- `scripts/run_training.sh` - Bash script to run training
- `requirements.txt` - Required packages

##  Tips & Tricks

1. **Start with base config**: Copy `configs/sft_diploy_8B.yaml` as starting point
2. **Change one thing at a time**: Easier to identify what works
3. **Document your changes**: Add comments in the YAML
4. **Use meaningful names**: `configs/sft_diploy_8B_high_lr.yaml` not `test1.yaml`
5. **Track results**: Use Weights & Biases to compare runs
6. **Keep successful configs**: Don't delete configs that worked well

##  Support

If you need help with configurations:
1. Check this README
2. Review the SETUP_INSTRUCTIONS.md in project root
3. Examine the training logs for clues
4. Compare with working configs

---

## Get in Touch with Maintainers
### Wildan: [![GitHub](https://img.shields.io/badge/GitHub-%23121011.svg?logo=github&logoColor=white)](https://github.com/wildanaziz) | [![Firefox](https://img.shields.io/badge/Firefox-FF7139?logo=firefoxbrowser&logoColor=white)](https://wildanaziz.vercel.app/) | [![Hugging Face](https://img.shields.io/badge/Hugging%20Face-FFD21E?logo=huggingface&logoColor=000)](https://huggingface.co/wildanaziz)
### Syafiq: [![GitHub](https://img.shields.io/badge/GitHub-%23121011.svg?logo=github&logoColor=white)](https://github.com/syafiqirz)
### Naufal: [![GitHub](https://img.shields.io/badge/GitHub-%23121011.svg?logo=github&logoColor=white)](https://github.com/NaufalArsa)

## Special Thanks to
1. **[TRL Official](https://huggingface.co/docs/trl/sft_trainer)**
2. **[SMOL Course](https://huggingface.co/learn/smol-course/unit0/1)**
3. **[Unsloth Official](https://docs.unsloth.ai/get-started/fine-tuning-llms-guide/lora-hyperparameters-guide)**
4. **[All papers in paper/](paper)**
