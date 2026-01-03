# Conversational Evaluation Guide

## Overview

The updated `run_evaluation.sh` script provides an **interactive** approach to evaluate your fine-tuned chatbot models using DeepEval framework with conversational and safety metrics.

### Key Features

**Interactive Model Selection** - Choose from local or HuggingFace models  
**Judge Model Selection** - Select from multiple OpenRouter API models  
**Dynamic Conversation Generation** - Creates realistic interview scenarios  
**8 Comprehensive Metrics** - Evaluates quality, role adherence, and safety  
**User-Friendly Interface** - Guided prompts for all configuration

---

## Quick Start

### 1. Interactive Mode (Recommended)

```bash
./scripts/run_evaluation.sh
```

You'll be prompted to:
1. **Select model to evaluate** (from list or enter custom path)
2. **Choose judge model** (6 options including GPT-4o-mini, Claude, etc.)
3. **Enter OpenRouter API key** (or use default)

### 2. Command-Line Mode

Specify all parameters upfront:

```bash
./scripts/run_evaluation.sh \
  -m wildanaziz/Diploy-8B-Base \
  -j openai/gpt-4o-mini \
  -k YOUR_OPENROUTER_API_KEY
```

---

## Evaluation Process

### What Happens During Evaluation?

1. **Model Loading**
   - Loads your model with 4-bit quantization (default)
   - Supports both local and HuggingFace models

2. **Conversation Generation**
   - Creates 3 interview scenarios (configurable with `--scenarios`)
   - Each scenario has 2+ conversation turns
   - Your model generates interviewer responses dynamically

3. **Conversational Metrics Evaluation** (5 metrics)
   - `TurnRelevancyMetric` - Response addresses user message
   - `KnowledgeRetentionMetric` - Remembers context across turns
   - `RoleAdherenceMetric` - Maintains interviewer persona
   - `ConversationCompletenessMetric` - Gathers all needed information
   - `TopicAdherenceMetric` - Stays focused on relevant topics

4. **Safety Metrics Evaluation** (3 metrics)
   - `ToxicityMetric` - No harmful/offensive language
   - `BiasMetric` - No discrimination
   - `HallucinationMetric` - No fabricated information

5. **Results**
   - Scores for all 8 metrics
   - Detailed logs saved to `evaluation_results/`
   - Results also available in DeepEval portal

---

## Available Options

### Model Selection

```bash
# Interactive selection from available models
./scripts/run_evaluation.sh

# Specify local model path
./scripts/run_evaluation.sh -m src/training/SFT-Qwen3-1.7B-LoRA-9GB-final

# Use HuggingFace model
./scripts/run_evaluation.sh -m wildanaziz/Diploy-8B-Base
```

### Judge Model Options

When prompted, select from:

| # | Model ID | Description | Cost |
|---|----------|-------------|------|
| 1 | `openai/gpt-4o-mini` | **Recommended** - Best balance | $$ |
| 2 | `openai/gpt-3.5-turbo` | Faster, cheaper | $ |
| 3 | `openai/gpt-4o` | Most accurate | $$$ |
| 4 | `anthropic/claude-3.5-sonnet` | High quality alternative | $$$ |
| 5 | `anthropic/claude-3-haiku` | Fast, affordable | $$ |
| 6 | Custom model ID | Enter your own | Varies |

### Additional Configuration

```bash
# Disable 4-bit quantization (requires more VRAM)
./scripts/run_evaluation.sh --no-4bit

# Adjust generation temperature
./scripts/run_evaluation.sh --temperature 0.7

# Change number of test scenarios
./scripts/run_evaluation.sh --scenarios 5

# Custom output directory
./scripts/run_evaluation.sh -o my_evaluation_results
```

---

## Understanding the Metrics

### Conversational Metrics (How well does it converse?)

1. **Turn Relevancy**
   - Does each response directly address the user's message?
   - Critical for maintaining natural conversation flow

2. **Knowledge Retention**
   - Does the bot remember information mentioned earlier?
   - Essential for interviews - recalls education, experience, skills

3. **Role Adherence**
   - Does it maintain its professional interviewer persona?
   - Prevents acting like candidate or giving irrelevant advice

4. **Conversation Completeness**
   - Did it gather all necessary information?
   - Ensures comprehensive data collection for assessment

5. **Topic Adherence**
   - Does it stay focused on relevant interview topics?
   - Avoids off-topic conversations (weather, sports, etc.)

### Safety Metrics (Is it safe and unbiased?)

6. **Toxicity** (Score: 0=safe, 1=toxic)
   - Detects harmful, offensive, or inappropriate language
   - Threshold: 0.5 (fails if >50% toxic)

7. **Bias** (Score: 0=unbiased, 1=biased)
   - Identifies discrimination based on gender, race, age, religion
   - Critical for fair hiring practices

8. **Hallucination** (Score: 0=accurate, 1=fabricated)
   - Ensures bot doesn't fabricate information
   - Only references actual candidate data

---

## Example Workflow

### Scenario 1: Quick Test

```bash
# Run with defaults - fully interactive
./scripts/run_evaluation.sh

# Follow prompts:
# 1. Select model: [0] for first local model or enter HF path
# 2. Judge model: [1] for GPT-4o-mini (recommended)
# 3. API key: [Enter] to use default or paste your key

# Wait 10-30 minutes for results
```

### Scenario 2: Production Evaluation

```bash
# Specify everything upfront
./scripts/run_evaluation.sh \
  -m wildanaziz/Diploy-8B-Base \
  -j openai/gpt-4o \
  --temperature 0.3 \
  --scenarios 5 \
  -o production_eval_$(date +%Y%m%d)
```

### Scenario 3: High-Memory GPU

```bash
# Disable quantization for better quality
./scripts/run_evaluation.sh \
  -m src/training/SFT-Qwen3-1.7B-LoRA-9GB-final \
  --no-4bit \
  -j openai/gpt-4o-mini
```

---

## Troubleshooting

### Issue: "No models found"

**Solution**: Either:
- Train a model first using `./scripts/run_training_unsloth.sh`
- Use a HuggingFace model path (e.g., `wildanaziz/Diploy-8B-Base`)

### Issue: "OpenRouter API key invalid"

**Solution**: Get your key from https://openrouter.ai/
```bash
export OPENROUTER_API_KEY="your-key-here"
./scripts/run_evaluation.sh
```

### Issue: "CUDA out of memory"

**Solution**: Use 4-bit quantization (default) or reduce scenarios:
```bash
./scripts/run_evaluation.sh --scenarios 2
```

### Issue: "Evaluation takes too long"

**Solutions**:
- Use faster judge model: `openai/gpt-3.5-turbo` or `anthropic/claude-3-haiku`
- Reduce scenarios: `--scenarios 2`
- Check GPU utilization: `nvidia-smi`

---

## Output Files

After evaluation, check `evaluation_results/` directory:

```
evaluation_results/
├── evaluation_20241226_143022.log  # Full execution log
└── (DeepEval stores results in their cloud portal)
```

**To view results:**
1. Check the log file for scores
2. Visit DeepEval portal: https://app.deepeval.ai/
3. Login and view detailed metrics, test cases, and comparisons

---

## Best Practices

### For Development

Use `gpt-4o-mini` as judge (balanced)  
Start with 2-3 scenarios for quick iteration  
Enable 4-bit quantization (default)  
Review logs after each run

### For Production/Research

Use `gpt-4o` or `claude-3.5-sonnet` for highest accuracy  
Increase to 5+ scenarios for comprehensive coverage  
Disable quantization if GPU allows (better quality)  
Document all configuration parameters  
Compare multiple models with same judge

### For Cost Optimization

Use `gpt-3.5-turbo` or `claude-3-haiku` as judge  
Limit scenarios to 2-3  
Enable 4-bit quantization  
Batch multiple evaluations in single session

---

## Advanced Usage

### Custom Scenarios

Edit the embedded Python script in `run_evaluation.sh` to add your own:

```python
user_scenarios = [
    {
        "name": "Your Custom Scenario",
        "messages": [
            "First user message...",
            "Second user message...",
        ]
    },
]
```

### Environment Variables

Set these before running:

```bash
export EVAL_MODEL_PATH="wildanaziz/Diploy-8B-Base"
export EVAL_JUDGE_MODEL="openai/gpt-4o-mini"
export OPENROUTER_API_KEY="your-key"
export EVAL_LOAD_4BIT="true"
export EVAL_TEMPERATURE="0.3"
export EVAL_NUM_SCENARIOS="3"

./scripts/run_evaluation.sh
```

---

## Comparison with Old Script

| Feature | Old Script | New Script |
|---------|-----------|-----------|
| Model Selection | Auto-detect only | **Interactive + Auto-detect + Custom** |
| Judge Model | Fixed (OpenAI) | **6+ options (OpenRouter)** |
| API Key | Manual export | **Interactive prompt** |
| Evaluation Type | Dataset-based | **Dynamic conversation generation** |
| Metrics | Generic LLM | **8 conversational + safety metrics** |
| User Experience | CLI-only | **Interactive + CLI** |

---

## Getting Help

```bash
# Show full help
./scripts/run_evaluation.sh --help

# Test GPU availability
nvidia-smi

# Check Python dependencies
python -c "import torch, transformers, deepeval; print('✓ All dependencies installed')"
```

---

## Next Steps

1. **Run your first evaluation** with interactive mode
2. **Review results** in DeepEval portal
3. **Tune your model** based on metric scores
4. **Re-evaluate** and compare improvements
5. **Deploy** when all metrics meet your thresholds

---

