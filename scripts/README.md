# Shell Scripts Package for Qwen3 Fine-tuning Pipeline

## Package Contents

This package contains all revised shell scripts for your `dtp-fine-tuning-research` project with correct path references and improved functionality.

### Shell Scripts (5 files)
1. **setup.sh** - Main setup and verification script
2. **run_training.sh** - Training pipeline script
3. **run_inference.sh** - Gradio inference server
4. **run_evaluation.sh** - OpenAI-based evaluation
5. **run_pipeline.sh** - Master orchestration script

### Documentation
See [**scripts_installation_guide.md**](docs/scripts_installation_guide.md)

## Quick Start

### Step 1: Backup Current Scripts
```bash
cd ~/dtp-fine-tuning-research
mkdir -p backup_scripts_$(date +%Y%m%d)
cp scripts/*.sh backup_scripts_$(date +%Y%m%d)/
```

### Step 2: Install New Scripts
```bash
#cpy downloaded scripts to your scripts directory
cp /path/to/downloaded/*.sh scripts/

#make executable
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

### run_training.sh
- Runs model training
- Validates config files
- Creates logs
- **Key Changes:** 
  - Default config multi-turn: `configs/sft_diploy_8B.yaml`
  - Default script: `src/training/train_unsloth_multi-turn.py`

### run_inference.sh
- Launches Gradio web interface
- Auto-detects latest model
- Displays model information
- **Key Changes:**
  - Default script: `src/inference/gradio_inference.py`
  - Searches models in root directory

### run_evaluation.sh
- Evaluates model with Openrouter models
- Generates reports with ConfidentAI using deepeval library
- **Key Changes:**
  - Default script: `src/eval/deepeval_my_model.py`
  - Auto-detects models from both locations

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
See [**scripts_installation_guide.md**](docs/scripts_installation_guide.md)

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
./scripts/run_inference.sh -m your-model-final

# Evaluation with deepeval
./scripts/run_evaluation.sh
```

### Pipeline Operations
```bash
# Complete pipeline
./scripts/run_pipeline.sh pipeline

# Individual steps
./scripts/run_pipeline.sh train
./scripts/run_pipeline.sh inference
./scripts/run_pipeline.sh evaluate
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
- `src/training/train_unsloth_multi-turn.py`
- `src/training/train_unsloth_single_turn.py`
- `src/inference/gradio_inference.py`
- `src/eval/deepeval_my_model.py`

### Environment Variables (needed for not harcode)
you can see the template on [**.env.template**](scripts/.env.template)

## Troubleshooting

### Scripts Won't Run
```bash
chmod +x scripts/*.sh
```

### "Must be run from project root"
```bash
cd ~/dtp-fine-tuning-research
#always run from here
```

### "Config/Script not found"
```bash
# verify files are in correct locations
ls configs/
ls src/training/*.py
ls src/inference/*.py
ls src/eval/*.py
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

# 3. Pipeline should show menu
./scripts/run_pipeline.sh
```

## Get in Touch with Maintainers
### Wildan: [![GitHub](https://img.shields.io/badge/GitHub-%23121011.svg?logo=github&logoColor=white)](https://github.com/wildanaziz) | [![Firefox](https://img.shields.io/badge/Firefox-FF7139?logo=firefoxbrowser&logoColor=white)](https://wildanaziz.vercel.app/) | [![Hugging Face](https://img.shields.io/badge/Hugging%20Face-FFD21E?logo=huggingface&logoColor=000)](https://huggingface.co/wildanaziz)
### Syafiq: [![GitHub](https://img.shields.io/badge/GitHub-%23121011.svg?logo=github&logoColor=white)](https://github.com/syafiqirz)
### Naufal: [![GitHub](https://img.shields.io/badge/GitHub-%23121011.svg?logo=github&logoColor=white)](https://github.com/NaufalArsa)

## Special thanks to:
1. **[The Linux Command Line: A Complete Introduction — William E. Shotts, Jr.](https://linuxcommand.org/index.php)**
2. **[Classic Shell Scripting — Arnold Robbins & Nelson H.F. Beebe](http://www.nylxs.com/docs/classicshellscripting.pdf)**
