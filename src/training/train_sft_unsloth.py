#!/usr/bin/env python3
"""
Unsloth SFT (Supervised Fine-Tuning) Training Script
Supports multiple model families with Unsloth optimization: Llama, Qwen, Gemma, Mistral, etc.

Features:
- Uses DataCollatorForSeq2Seq + train_on_responses_only for instruction masking
- Chat template detection from config YAML
- BLEU and ROUGE evaluation after training
- WandB artifacts upload
- Configuration-driven training
"""

import os
import sys
import torch
import yaml
import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime

# Unsloth imports
try:
    from unsloth import FastLanguageModel, is_bfloat16_supported
    from unsloth.chat_templates import get_chat_template, train_on_responses_only
except ImportError:
    print("❌ Error: Unsloth library not installed!")
    print("Install with: pip install unsloth")
    print("Or with conda: conda install unsloth -c conda-forge")
    sys.exit(1)

from datasets import load_dataset, Dataset
from trl import SFTTrainer
from transformers import (
    TrainingArguments,
    DataCollatorForSeq2Seq,
    TrainerCallback,
    EarlyStoppingCallback
)
import wandb

# Evaluation imports
try:
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    from rouge_score import rouge_scorer
    import nltk
    # Download required NLTK data
    try:
        nltk.download('punkt', quiet=True)
        nltk.download('punkt_tab', quiet=True)
    except:
        pass
    EVAL_AVAILABLE = True
except ImportError:
    print("⚠️  Warning: Evaluation libraries not fully installed")
    print("Install with: pip install rouge-score nltk")
    EVAL_AVAILABLE = False


# ============================================================================
# Model Family Configuration Registry
# ============================================================================

MODEL_FAMILIES = {
    'llama': {
        'patterns': ['llama', 'llama-2', 'llama-3', 'codellama'],
        'chat_template': 'llama-3.1',
        'instruction_part': '<|start_header_id|>user<|end_header_id|>\n\n',
        'response_part': '<|start_header_id|>assistant<|end_header_id|>\n\n',
        'eos_token': '<|eot_id|>',
    },
    'qwen': {
        'patterns': ['qwen', 'qwen2', 'qwen2.5', 'qwen3'],
        'chat_template': 'qwen-2.5',
        'instruction_part': '<|im_start|>user\n',
        'response_part': '<|im_start|>assistant\n',
        'eos_token': '<|im_end|>',
    },
    'mistral': {
        'patterns': ['mistral', 'mixtral'],
        'chat_template': 'mistral',
        'instruction_part': '[INST]',
        'response_part': '[/INST]',
        'eos_token': '</s>',
    },
    'gemma': {
        'patterns': ['gemma'],
        'chat_template': 'gemma',
        'instruction_part': '<start_of_turn>user\n',
        'response_part': '<start_of_turn>model\n',
        'eos_token': '<end_of_turn>',
    },
    'phi': {
        'patterns': ['phi-2', 'phi-3'],
        'chat_template': 'phi-3',
        'instruction_part': '<|user|>\n',
        'response_part': '<|assistant|>\n',
        'eos_token': '<|end|>',
    },
}


def detect_model_family(model_name: str) -> Optional[str]:
    """
    Detect the model family from model name.

    Args:
        model_name: HuggingFace model identifier

    Returns:
        Model family name or None if not recognized
    """
    model_name_lower = model_name.lower()

    for family, config in MODEL_FAMILIES.items():
        for pattern in config['patterns']:
            if pattern in model_name_lower:
                return family

    return None


# ============================================================================
# Custom Callbacks
# ============================================================================

class CustomLoggingCallback(TrainerCallback):
    """Custom callback for additional logging and monitoring"""

    def on_log(self, args, state, control, logs=None, **kwargs):
        """Called when logging occurs"""
        if logs is not None:
            if "loss" in logs:
                print(f"Step {state.global_step}: Training Loss = {logs['loss']:.4f}")
            if "eval_loss" in logs:
                print(f"Step {state.global_step}: Eval Loss = {logs['eval_loss']:.4f}")

    def on_epoch_end(self, args, state, control, **kwargs):
        """Called at the end of each epoch"""
        print(f"\n{'='*50}")
        print(f"Epoch {state.epoch} completed!")
        print(f"{'='*50}\n")


class MemoryMonitorCallback(TrainerCallback):
    """Callback to monitor GPU memory usage"""

    def __init__(self, log_every_n_steps=100):
        self.log_every_n_steps = log_every_n_steps

    def on_step_end(self, args, state, control, **kwargs):
        """Called at the end of each training step"""
        if state.global_step % self.log_every_n_steps == 0:
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated() / 1024**3  # GB
                reserved = torch.cuda.memory_reserved() / 1024**3    # GB
                print(f"GPU Memory - Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB")


class SaveBestModelCallback(TrainerCallback):
    """Callback to save the best model based on custom criteria"""

    def __init__(self):
        self.best_metric = float('inf')

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        """Called after evaluation"""
        if metrics is not None and "eval_loss" in metrics:
            current_metric = metrics["eval_loss"]
            if current_metric < self.best_metric:
                self.best_metric = current_metric
                print(f"\n🎯 New best model! Eval Loss: {current_metric:.4f}")


# ============================================================================
# Helper Functions
# ============================================================================

def save_training_info(config: Dict, output_dir: str, model_family: str):
    """Save training configuration and metadata for later inference"""
    training_info = {
        "model_name": config['model']['name'],
        "model_family": model_family,
        "lora_config": config['lora'],
        "tokenizer_config": config['tokenizer'],
        "chat_template": config.get('chat_template', {}),
        "training_completed": datetime.now().isoformat(),
        "dataset_info": {
            "name": config['dataset']['name'],
            "max_length": config['dataset']['max_length']
        },
        "training_framework": "unsloth"
    }

    info_path = Path(output_dir) / "training_info.json"
    with open(info_path, 'w') as f:
        json.dump(training_info, f, indent=2)
    print(f"📄 Training info saved to: {info_path}")


def upload_model_to_wandb(config: Dict, model_dir: str, model_family: str,
                         artifact_name: str = None, artifact_type: str = "model"):
    """
    Upload trained model to W&B as an artifact.

    Args:
        config: Configuration dictionary
        model_dir: Directory containing the model to upload
        model_family: Detected model family
        artifact_name: Optional custom artifact name
        artifact_type: Type of artifact (default: "model")
    """
    # Check if W&B is enabled
    if config.get('training', {}).get('report_to', 'none') == 'none':
        print("⚠️  W&B logging disabled, skipping artifact upload")
        return

    if not os.environ.get('WANDB_API_KEY'):
        print("⚠️  WANDB_API_KEY not set, skipping artifact upload")
        return

    try:
        # Ensure wandb is initialized
        if wandb.run is None:
            print("⚠️  No active W&B run, skipping artifact upload")
            return

        # Generate artifact name if not provided
        if artifact_name is None:
            model_name = config['model']['name'].split('/')[-1]
            dataset_name = config['dataset']['name'].split('/')[-1]
            artifact_name = f"{model_name}-{dataset_name}-unsloth-lora"

        print(f"\n📤 Uploading model artifact to W&B...")
        print(f"   Artifact name: {artifact_name}")
        print(f"   Model directory: {model_dir}")

        # Create artifact
        artifact = wandb.Artifact(
            name=artifact_name,
            type=artifact_type,
            description=f"Fine-tuned {model_family or 'LLM'} model with Unsloth LoRA adapters",
            metadata={
                "model_name": config['model']['name'],
                "model_family": model_family or "unknown",
                "dataset": config['dataset']['name'],
                "lora_r": config['lora']['r'],
                "lora_alpha": config['lora']['lora_alpha'],
                "max_length": config['dataset']['max_length'],
                "learning_rate": config['training']['learning_rate'],
                "num_epochs": config['training']['num_train_epochs'],
                "batch_size": config['training']['per_device_train_batch_size'],
                "gradient_accumulation": config['training']['gradient_accumulation_steps'],
                "training_framework": "unsloth",
                "training_completed": datetime.now().isoformat()
            }
        )

        # Add model directory to artifact
        artifact.add_dir(model_dir)

        # Log artifact to W&B
        wandb.log_artifact(artifact)

        print(f"✅ Model artifact uploaded successfully!")
        print(f"   View at: {wandb.run.get_url()}")

    except Exception as e:
        print(f"⚠️  Failed to upload model artifact: {e}")


# ============================================================================
# Dataset Preparation
# ============================================================================

def load_and_prepare_dataset(config: Dict) -> Tuple[Dataset, Dataset]:
    """Load dataset from HuggingFace and prepare train/eval splits"""
    dataset_name = config['dataset']['name']
    split = config['dataset'].get('split', 'train')
    test_size = config['dataset'].get('test_size', 0.1)
    seed = config['dataset'].get('seed', 42)

    print(f"📊 Loading dataset: {dataset_name}")

    # Load dataset
    dataset = load_dataset(dataset_name, split=split)

    # Split into train and eval
    if test_size > 0:
        dataset_split = dataset.train_test_split(test_size=test_size, seed=seed)
        train_dataset = dataset_split['train']
        eval_dataset = dataset_split['test']
    else:
        train_dataset = dataset
        eval_dataset = None

    return train_dataset, eval_dataset


def format_dataset_with_chat_template(dataset: Dataset, tokenizer, config: Dict):
    """
    Format dataset using tokenizer's chat template.
    Converts various dataset formats to text format with applied chat template.
    """
    def format_example(example):
        messages = []

        # Extract messages from various formats
        if 'messages' in example:
            messages = list(example['messages'])
        elif 'conversation' in example:
            messages = list(example['conversation'])
        elif 'conversations' in example:
            messages = list(example['conversations'])
        elif 'instruction' in example and 'output' in example:
            messages = [
                {"role": "user", "content": example['instruction']},
                {"role": "assistant", "content": example['output']}
            ]
        elif 'input' in example and 'output' in example:
            messages = [
                {"role": "user", "content": example['input']},
                {"role": "assistant", "content": example['output']}
            ]
        elif 'text' in example:
            # Already formatted
            return {"text": example['text']}

        # Handle separate system field
        if 'system' in example and example['system']:
            system_msg = {"role": "system", "content": example['system']}
            if not messages or messages[0].get('role') != 'system':
                messages.insert(0, system_msg)
        elif config.get('chat_template', {}).get('use_system_message', False):
            system_msg = {"role": "system", "content": config['chat_template'].get('system_message', '')}
            if not messages or messages[0].get('role') != 'system':
                messages.insert(0, system_msg)

        # Normalize role names
        normalized_messages = []
        for msg in messages:
            if isinstance(msg, dict):
                role = msg.get("role", "user")
                content = msg.get("content", msg.get("value", ""))
            else:
                role = msg[0] if len(msg) > 0 else "user"
                content = msg[1] if len(msg) > 1 else ""

            # Normalize role names
            if role in ["human", "Human"]:
                role = "user"
            elif role in ["ai", "bot", "gpt", "Assistant"]:
                role = "assistant"

            normalized_messages.append({"role": role, "content": content})

        # Apply chat template
        try:
            text = tokenizer.apply_chat_template(
                normalized_messages,
                tokenize=False,
                add_generation_prompt=False
            )
            return {"text": text}
        except Exception as e:
            print(f"⚠️  Error applying chat template: {e}")
            raise

    # Apply formatting
    dataset = dataset.map(
        format_example,
        remove_columns=dataset.column_names,
        desc="Formatting dataset with chat template"
    )

    return dataset


# ============================================================================
# Chat Template Setup
# ============================================================================

def setup_chat_template(tokenizer, model_name: str, config: Dict, model_family: str):
    """
    Setup chat template for the tokenizer using Unsloth's get_chat_template.

    Args:
        tokenizer: Tokenizer instance
        model_name: Model identifier
        config: Configuration dictionary
        model_family: Detected model family

    Returns:
        Tokenizer with chat template applied
    """
    # Check if custom template in config
    if 'chat_template' in config and 'template_string' in config['chat_template']:
        tokenizer.chat_template = config['chat_template']['template_string']
        print("✓ Applied custom chat template from config")
        return tokenizer

    # Use Unsloth's chat template for model family
    if model_family and model_family in MODEL_FAMILIES:
        template_name = MODEL_FAMILIES[model_family]['chat_template']
        print(f"🔧 Applying Unsloth chat template: {template_name}")

        try:
            tokenizer = get_chat_template(
                tokenizer,
                chat_template=template_name,
            )
            print(f"✓ Applied {template_name} chat template")
        except Exception as e:
            print(f"⚠️  Could not apply Unsloth template: {e}")
            print("   Using tokenizer's default template")

    return tokenizer


# ============================================================================
# Evaluation Functions
# ============================================================================

def calculate_bleu_rouge(predictions: List[str], references: List[str]) -> Dict[str, float]:
    """
    Calculate BLEU and ROUGE scores for model predictions.

    Args:
        predictions: List of generated texts
        references: List of reference texts

    Returns:
        Dictionary with BLEU and ROUGE scores
    """
    if not EVAL_AVAILABLE:
        print("⚠️  Evaluation libraries not available, skipping BLEU/ROUGE calculation")
        return {}

    print("\n📊 Calculating BLEU and ROUGE scores...")

    # BLEU scores
    bleu_scores = []
    smoothing = SmoothingFunction()

    for pred, ref in zip(predictions, references):
        # Tokenize
        pred_tokens = pred.split()
        ref_tokens = ref.split()

        # Calculate BLEU
        try:
            bleu = sentence_bleu(
                [ref_tokens],
                pred_tokens,
                smoothing_function=smoothing.method1
            )
            bleu_scores.append(bleu)
        except:
            bleu_scores.append(0.0)

    avg_bleu = sum(bleu_scores) / len(bleu_scores) if bleu_scores else 0.0

    # ROUGE scores
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    rouge1_scores = []
    rouge2_scores = []
    rougeL_scores = []

    for pred, ref in zip(predictions, references):
        scores = scorer.score(ref, pred)
        rouge1_scores.append(scores['rouge1'].fmeasure)
        rouge2_scores.append(scores['rouge2'].fmeasure)
        rougeL_scores.append(scores['rougeL'].fmeasure)

    avg_rouge1 = sum(rouge1_scores) / len(rouge1_scores) if rouge1_scores else 0.0
    avg_rouge2 = sum(rouge2_scores) / len(rouge2_scores) if rouge2_scores else 0.0
    avg_rougeL = sum(rougeL_scores) / len(rougeL_scores) if rougeL_scores else 0.0

    results = {
        'bleu': avg_bleu,
        'rouge1': avg_rouge1,
        'rouge2': avg_rouge2,
        'rougeL': avg_rougeL
    }

    print(f"✓ BLEU Score:   {avg_bleu:.4f}")
    print(f"✓ ROUGE-1:      {avg_rouge1:.4f}")
    print(f"✓ ROUGE-2:      {avg_rouge2:.4f}")
    print(f"✓ ROUGE-L:      {avg_rougeL:.4f}")

    return results


def evaluate_model(model, tokenizer, eval_dataset: Dataset, config: Dict) -> Dict[str, float]:
    """
    Evaluate the trained model using BLEU and ROUGE metrics.

    Args:
        model: Trained model
        tokenizer: Tokenizer
        eval_dataset: Evaluation dataset (text format)
        config: Configuration dictionary

    Returns:
        Dictionary with evaluation metrics
    """
    if not EVAL_AVAILABLE:
        print("⚠️  Evaluation libraries not available, skipping evaluation")
        return {}

    print("\n" + "="*70)
    print("🎯 EVALUATING MODEL WITH BLEU AND ROUGE")
    print("="*70)

    # Get number of samples to evaluate
    num_samples = min(
        config.get('evaluation', {}).get('test_samples', 100),
        len(eval_dataset)
    )

    print(f"📝 Evaluating on {num_samples} samples...")

    # Prepare model for inference
    FastLanguageModel.for_inference(model)

    # Generate predictions
    predictions = []
    references = []

    for i in range(num_samples):
        example = eval_dataset[i]
        text = example['text']

        # Split into input and reference
        # Find the last occurrence of assistant response marker
        model_family = detect_model_family(config['model']['name'])
        if model_family and model_family in MODEL_FAMILIES:
            response_marker = MODEL_FAMILIES[model_family]['response_part']
        else:
            response_marker = "assistant"

        # Find last assistant response
        parts = text.split(response_marker)
        if len(parts) >= 2:
            input_text = response_marker.join(parts[:-1]) + response_marker
            reference_text = parts[-1].strip()

            # Remove EOS token from reference
            if model_family and model_family in MODEL_FAMILIES:
                eos_token = MODEL_FAMILIES[model_family]['eos_token']
                reference_text = reference_text.replace(eos_token, '').strip()
        else:
            # Fallback: use first 50% as input
            mid = len(text) // 2
            input_text = text[:mid]
            reference_text = text[mid:]

        # Tokenize input
        inputs = tokenizer(
            input_text,
            return_tensors="pt",
            truncation=True,
            max_length=config['dataset']['max_length']
        ).to(model.device)

        # Generate
        outputs = model.generate(
            **inputs,
            max_new_tokens=config.get('inference', {}).get('max_new_tokens', 512),
            temperature=config.get('inference', {}).get('temperature', 0.7),
            top_p=config.get('inference', {}).get('top_p', 0.95),
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

        # Decode
        generated_text = tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        )

        predictions.append(generated_text.strip())
        references.append(reference_text.strip())

        # Progress indicator
        if (i + 1) % 10 == 0:
            print(f"   Processed {i + 1}/{num_samples} samples...")

    # Calculate metrics
    metrics = calculate_bleu_rouge(predictions, references)

    # Save predictions if requested
    if config.get('evaluation', {}).get('save_predictions', False):
        output_dir = Path(config.get('evaluation', {}).get('output_dir', './evaluation_results'))
        output_dir.mkdir(parents=True, exist_ok=True)

        predictions_file = output_dir / f"predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(predictions_file, 'w', encoding='utf-8') as f:
            json.dump({
                'predictions': predictions,
                'references': references,
                'metrics': metrics
            }, f, indent=2, ensure_ascii=False)

        print(f"\n💾 Predictions saved to: {predictions_file}")

    # Log to WandB if enabled
    if wandb.run is not None and metrics:
        wandb.log({
            "eval/bleu": metrics.get('bleu', 0),
            "eval/rouge1": metrics.get('rouge1', 0),
            "eval/rouge2": metrics.get('rouge2', 0),
            "eval/rougeL": metrics.get('rougeL', 0)
        })

    return metrics


# ============================================================================
# Main Training Function
# ============================================================================

def main(config_path: str):
    """Main training pipeline"""

    # Load configuration
    print("="*70)
    print("🚀 UNSLOTH SFT TRAINING PIPELINE")
    print("="*70 + "\n")

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    print(f"📄 Configuration loaded from: {config_path}\n")

    # Detect model family
    model_name = config['model']['name']
    model_family = detect_model_family(model_name)

    if model_family:
        print(f"🔍 Detected model family: {model_family}")
    else:
        print(f"⚠️  Unknown model family for: {model_name}")
        print("   Will use default settings\n")

    # Initialize W&B
    if config.get('training', {}).get('report_to') == 'wandb':
        if os.environ.get('WANDB_API_KEY'):
            wandb_config = config.get('wandb', {})
            wandb.init(
                entity=wandb_config.get('entity'),
                project=wandb_config.get('project'),
                name=config['training'].get('run_name'),
                tags=wandb_config.get('tags', []) + ['unsloth'],
                notes=wandb_config.get('notes', ''),
                config=config
            )
            print("✓ Weights & Biases initialized\n")
        else:
            print("⚠️  WANDB_API_KEY not found. WandB logging disabled.\n")
            config['training']['report_to'] = 'none'

    # Load model and tokenizer with Unsloth
    print("🔧 Loading model with Unsloth optimization...")

    max_seq_length = config['dataset']['max_length']
    dtype = None  # Auto-detect
    load_in_4bit = config.get('quantization', {}).get('load_in_4bit', True)

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        dtype=dtype,
        load_in_4bit=load_in_4bit,
        trust_remote_code=config['model'].get('trust_remote_code', False),
        use_gradient_checkpointing=False,
    )

    print(f"✓ Model loaded: {model_name}")
    print(f"✓ Max sequence length: {max_seq_length}")
    print(f"✓ 4-bit quantization: {load_in_4bit}\n")

    # Setup LoRA with Unsloth
    print("🔧 Configuring LoRA with Unsloth...")

    lora_config = config['lora']
    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_config['r'],
        lora_alpha=lora_config['lora_alpha'],
        lora_dropout=lora_config['lora_dropout'],
        target_modules=lora_config['target_modules'],
        bias=lora_config.get('bias', 'none'),
        use_gradient_checkpointing=False,  # Unsloth's optimized gradient checkpointing
        random_state=config['dataset'].get('seed', 42),
        use_rslora=False,
        loftq_config=None,
    )

    print(f"✓ LoRA configured with r={lora_config['r']}, alpha={lora_config['lora_alpha']}")
    print(f"✓ Target modules: {', '.join(lora_config['target_modules'])}\n")

    # Setup tokenizer and chat template
    print("🔧 Setting up tokenizer and chat template...")

    # Set padding side
    padding_side = config.get('tokenizer', {}).get('padding_side', 'left')
    tokenizer.padding_side = padding_side

    # Ensure pad token is set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Apply chat template
    tokenizer = setup_chat_template(tokenizer, model_name, config, model_family)

    print(f"✓ Tokenizer configured (padding_side: {padding_side})\n")

    # Load and prepare dataset
    print("📊 Loading and preparing dataset...")
    train_dataset, eval_dataset = load_and_prepare_dataset(config)

    print(f"✓ Train samples: {len(train_dataset):,}")
    if eval_dataset:
        print(f"✓ Eval samples: {len(eval_dataset):,}\n")

    # Format dataset with chat template
    print("💬 Formatting datasets with chat template...")
    train_dataset = format_dataset_with_chat_template(train_dataset, tokenizer, config)
    if eval_dataset:
        eval_dataset = format_dataset_with_chat_template(eval_dataset, tokenizer, config)

    # Print sample
    print("\n📝 Sample formatted text:")
    print("-" * 70)
    sample_text = train_dataset[0]['text']
    # Print first 500 chars
    print(sample_text[:500] + "..." if len(sample_text) > 500 else sample_text)
    print("-" * 70 + "\n")

    # Check if assistant-only training is enabled
    assistant_only = config.get('training', {}).get('assistant_only_loss', True)

    if assistant_only:
        print("🎭 Instruction Masking: ENABLED")
        print("   ✓ Will train ONLY on assistant responses (user/system masked)")
        print("   ✓ Using Unsloth's train_on_responses_only wrapper")

        # Get instruction and response parts
        if model_family and model_family in MODEL_FAMILIES:
            instruction_part = MODEL_FAMILIES[model_family]['instruction_part']
            response_part = MODEL_FAMILIES[model_family]['response_part']
        else:
            # Use from config or defaults
            instruction_part = config.get('training', {}).get('instruction_part', 'user')
            response_part = config.get('training', {}).get('response_part', 'assistant')

        print(f"   ✓ Instruction part: {repr(instruction_part)}")
        print(f"   ✓ Response part: {repr(response_part)}\n")
    else:
        print("🎭 Instruction Masking: DISABLED")
        print("   Training on full conversations\n")

    # Create training arguments
    print("🔧 Creating training configuration...")

    train_cfg = config['training']
    output_dir = train_cfg['output_dir']

    training_args = TrainingArguments(
        # Output
        output_dir=output_dir,
        run_name=train_cfg.get('run_name'),

        # Batch configuration
        per_device_train_batch_size=train_cfg['per_device_train_batch_size'],
        per_device_eval_batch_size=train_cfg.get('per_device_eval_batch_size', 1),
        gradient_accumulation_steps=train_cfg['gradient_accumulation_steps'],

        # Training parameters
        num_train_epochs=train_cfg['num_train_epochs'],
        learning_rate=train_cfg['learning_rate'],
        warmup_ratio=train_cfg.get('warmup_ratio', 0.03),
        warmup_steps=train_cfg.get('warmup_steps', 0),
        max_steps=train_cfg.get('max_steps', -1),

        # Optimizer
        optim=train_cfg.get('optim', 'adamw_8bit'),
        weight_decay=train_cfg.get('weight_decay', 0.01),

        # Scheduler
        lr_scheduler_type=train_cfg.get('lr_scheduler_type', 'linear'),

        # Precision
        bf16=is_bfloat16_supported(),
        fp16=not is_bfloat16_supported(),

        # Logging & Evaluation
        logging_steps=train_cfg.get('logging_steps', 1),
        eval_strategy=train_cfg.get('eval_strategy', 'steps') if eval_dataset else 'no',
        eval_steps=train_cfg.get('eval_steps', 100) if eval_dataset else None,
        save_strategy=train_cfg.get('save_strategy', 'steps'),
        save_steps=train_cfg.get('save_steps', 100),
        save_total_limit=train_cfg.get('save_total_limit', 2),

        # Model selection
        load_best_model_at_end=train_cfg.get('load_best_model_at_end', True) if eval_dataset else False,
        metric_for_best_model=train_cfg.get('metric_for_best_model', 'eval_loss'),
        greater_is_better=train_cfg.get('greater_is_better', False),

        # Misc
        seed=config['dataset'].get('seed', 42),
        report_to=train_cfg.get('report_to', 'none'),
    )

    print("✓ Training arguments created\n")

    # Setup callbacks
    callbacks = []

    if config.get('callbacks', {}).get('early_stopping', {}).get('enabled') and eval_dataset:
        early_stopping_patience = config['callbacks']['early_stopping']['patience']
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=early_stopping_patience))
        print(f"✓ Early stopping enabled (patience={early_stopping_patience})")

    if config.get('callbacks', {}).get('custom_logging', {}).get('enabled'):
        callbacks.append(CustomLoggingCallback())
        print("✓ Custom logging callback enabled")

    if config.get('callbacks', {}).get('memory_monitor', {}).get('enabled'):
        log_every = config['callbacks']['memory_monitor'].get('log_every_n_steps', 100)
        callbacks.append(MemoryMonitorCallback(log_every_n_steps=log_every))
        print(f"✓ Memory monitor callback enabled (every {log_every} steps)")

    if config.get('callbacks', {}).get('save_best_model', {}).get('enabled'):
        callbacks.append(SaveBestModelCallback())
        print("✓ Save best model callback enabled")

    print()

    # Create SFTTrainer with DataCollatorForSeq2Seq
    print("🔧 Creating Unsloth SFTTrainer with DataCollatorForSeq2Seq...")

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer),
        dataset_num_proc=train_cfg.get('dataset_num_proc', 2),
        packing=train_cfg.get('packing', False),
        callbacks=callbacks,
    )

    print("✓ SFTTrainer created with DataCollatorForSeq2Seq\n")

    # Apply train_on_responses_only if assistant-only training
    if assistant_only:
        if model_family and model_family in MODEL_FAMILIES:
            instruction_part = MODEL_FAMILIES[model_family]['instruction_part']
            response_part = MODEL_FAMILIES[model_family]['response_part']

            try:
                trainer = train_on_responses_only(
                    trainer,
                    instruction_part=instruction_part,
                    response_part=response_part,
                )
                print("✓ Applied train_on_responses_only for instruction masking")
                print(f"   Instruction part: {repr(instruction_part)}")
                print(f"   Response part: {repr(response_part)}\n")
            except Exception as e:
                print(f"⚠️  Could not apply train_on_responses_only: {e}")
                print("   Training will proceed on full conversations\n")

    # Print training summary
    print("="*70)
    print("🎯 TRAINING CONFIGURATION SUMMARY")
    print("="*70)
    print(f"Model:                    {model_name}")
    print(f"Model Family:             {model_family or 'unknown'}")
    print(f"Dataset:                  {config['dataset']['name']}")
    print(f"Training Framework:       Unsloth + DataCollatorForSeq2Seq")
    print(f"Max Sequence Length:      {max_seq_length}")
    print(f"LoRA Rank:                {lora_config['r']}")
    print(f"LoRA Alpha:               {lora_config['lora_alpha']}")
    print(f"Batch Size:               {train_cfg['per_device_train_batch_size']}")
    print(f"Gradient Accumulation:    {train_cfg['gradient_accumulation_steps']}")
    print(f"Learning Rate:            {train_cfg['learning_rate']}")
    print(f"Epochs:                   {train_cfg['num_train_epochs']}")
    print(f"Max Steps:                {train_cfg.get('max_steps', -1)}")

    if assistant_only:
        print(f"Instruction Masking:      ✓ ENABLED (train_on_responses_only)")
        print(f"Training Strategy:        Train ONLY on assistant responses")
    else:
        print(f"Instruction Masking:      ✗ Disabled")
        print(f"Training Strategy:        Train on full conversations")

    if config.get('callbacks', {}).get('early_stopping', {}).get('enabled'):
        print(f"Early Stopping Patience:  {config['callbacks']['early_stopping']['patience']}")
    print("="*70 + "\n")

    # Train
    try:
        print("🏋️ Starting training with Unsloth...\n")
        trainer_stats = trainer.train()
        print(f"\n✓ Training completed!")
        print(f"   Training loss: {trainer_stats.training_loss:.4f}")
    except torch.cuda.OutOfMemoryError:
        print("\n❌ Out of Memory Error!")
        print("\n💡 Suggestions:")
        print(f"  • Reduce max_length in config (currently: {config['dataset']['max_length']})")
        print(f"  • Reduce LoRA r value (currently: {config['lora']['r']})")
        print(f"  • Increase gradient_accumulation_steps (currently: {config['training']['gradient_accumulation_steps']})")
        print(f"  • Reduce batch size (currently: {config['training']['per_device_train_batch_size']})")
        raise
    except Exception as e:
        print(f"\n❌ Training failed with error: {e}")
        raise

    # Evaluate model with BLEU and ROUGE
    if eval_dataset and config.get('evaluation', {}).get('metrics'):
        try:
            eval_metrics = evaluate_model(model, tokenizer, eval_dataset, config)

            if eval_metrics:
                # Add to training summary
                print("\n📊 Evaluation Metrics:")
                print(f"   BLEU:    {eval_metrics.get('bleu', 0):.4f}")
                print(f"   ROUGE-1: {eval_metrics.get('rouge1', 0):.4f}")
                print(f"   ROUGE-2: {eval_metrics.get('rouge2', 0):.4f}")
                print(f"   ROUGE-L: {eval_metrics.get('rougeL', 0):.4f}")
        except Exception as e:
            print(f"\n⚠️  Evaluation failed: {e}")

    # Save final model
    print("\n" + "="*70)
    print("💾 Training completed! Saving final model...")
    print("="*70 + "\n")

    final_model_dir = config['paths']['final_model_dir']

    # Save model and tokenizer
    model.save_pretrained(final_model_dir)
    tokenizer.save_pretrained(final_model_dir)

    print(f"✓ Model saved to: {final_model_dir}")

    # Save training info for later inference
    save_training_info(config, final_model_dir, model_family or 'unknown')

    # Upload model artifact to W&B
    upload_model_to_wandb(config, final_model_dir, model_family or 'unknown')

    # Also save merged model if specified
    if config.get('save_merged_model', False):
        merged_dir = Path(final_model_dir).parent / f"{Path(final_model_dir).name}_merged"
        print(f"\n🔗 Saving merged model to: {merged_dir}")

        # Merge LoRA weights with base model using Unsloth
        model.save_pretrained_merged(
            str(merged_dir),
            tokenizer,
            save_method="merged_16bit",  # Options: "merged_16bit", "merged_4bit", "lora"
        )

        print(f"✓ Merged model saved to: {merged_dir}")

        # Upload merged model to W&B
        upload_model_to_wandb(
            config,
            str(merged_dir),
            model_family or 'unknown',
            artifact_name=f"{config['model']['name'].split('/')[-1]}-{config['dataset']['name'].split('/')[-1]}-unsloth-merged",
            artifact_type="merged-model"
        )

    # Print training summary
    if trainer.state.best_metric is not None:
        print(f"\n🏆 Best eval loss achieved: {trainer.state.best_metric:.4f}")
        if trainer.state.best_model_checkpoint:
            print(f"📁 Best model checkpoint: {trainer.state.best_model_checkpoint}")

    print("\n✅ Training pipeline completed successfully!")
    print(f"📁 Model saved to: {final_model_dir}")

    # Final memory check
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"\n💾 Final GPU Memory - Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB")

    # Close wandb
    if wandb.run is not None:
        wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Unsloth SFT Training Script - Optimized for Llama, Qwen, Gemma, Mistral, and more"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the YAML configuration file"
    )

    args = parser.parse_args()

    # Verify config file exists
    if not Path(args.config).exists():
        print(f"❌ Error: Configuration file not found at {args.config}")
        sys.exit(1)

    main(args.config)
