#!/usr/bin/env python3
"""
Generic SFT (Supervised Fine-Tuning) Training Script
Supports multiple model families: Llama, Qwen, Mistral, Gemma, etc.
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
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainerCallback,
    EarlyStoppingCallback,
    BitsAndBytesConfig
)
from datasets import load_dataset, Dataset
from trl import SFTTrainer, SFTConfig, DataCollatorForCompletionOnlyLM
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel
import wandb
from datetime import datetime


# ============================================================================
# Model Family Configuration Registry
# ============================================================================

MODEL_FAMILIES = {
    'llama': {
        'patterns': ['llama', 'llama-2', 'llama-3', 'codellama'],
        'response_template': '<|start_header_id|>assistant<|end_header_id|>\n\n',
        'default_padding_side': 'right',
        'default_chat_template': None,  # Use tokenizer's built-in
    },
    'qwen': {
        'patterns': ['qwen', 'qwen2', 'qwen2.5', 'qwen3'],
        'response_template': '<|im_start|>assistant\n',
        'default_padding_side': 'left',
        'default_chat_template': None,  # Use tokenizer's built-in
    },
    'mistral': {
        'patterns': ['mistral', 'mixtral'],
        'response_template': '[/INST]',
        'default_padding_side': 'left',
        'default_chat_template': None,
    },
    'gemma': {
        'patterns': ['gemma'],
        'response_template': '<start_of_turn>model\n',
        'default_padding_side': 'right',
        'default_chat_template': None,
    },
    'phi': {
        'patterns': ['phi-2', 'phi-3'],
        'response_template': '<|assistant|>\n',
        'default_padding_side': 'left',
        'default_chat_template': None,
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


def get_response_template(model_name: str, tokenizer: AutoTokenizer, config: Dict) -> str:
    """
    Auto-detect or retrieve the response template for instruction masking.

    Args:
        model_name: Model identifier
        tokenizer: Tokenizer instance
        config: Configuration dictionary

    Returns:
        Response template string
    """
    # Check if manually specified in config
    if 'response_template' in config.get('training', {}):
        template = config['training']['response_template']
        print(f"Using manually specified response template: {repr(template)}")
        return template

    # Auto-detect based on model family
    family = detect_model_family(model_name)
    if family and family in MODEL_FAMILIES:
        template = MODEL_FAMILIES[family]['response_template']
        print(f"Auto-detected model family '{family}' with response template: {repr(template)}")
        return template

    # Fallback: try to infer from tokenizer's chat template
    if hasattr(tokenizer, 'chat_template') and tokenizer.chat_template:
        chat_template = tokenizer.chat_template

        # Common patterns to search for
        assistant_patterns = [
            r'<\|start_header_id\|>assistant<\|end_header_id\|>',
            r'<\|im_start\|>assistant',
            r'\[/INST\]',
            r'<start_of_turn>model',
            r'<\|assistant\|>',
        ]

        for pattern in assistant_patterns:
            if re.search(pattern, chat_template):
                # Extract the matched pattern and add newlines if needed
                match = re.search(pattern, chat_template)
                template = match.group(0)
                if not template.endswith('\n'):
                    template += '\n'
                print(f"Inferred response template from chat_template: {repr(template)}")
                return template

    # Final fallback
    print("WARNING: Could not auto-detect response template. Using generic fallback.")
    print("Consider specifying 'response_template' in your config under 'training' section.")
    return "Assistant:"


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
# Configuration and Setup Functions
# ============================================================================

def load_config(config_path: str) -> Dict:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


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
        }
    }

    info_path = Path(output_dir) / "training_info.json"
    with open(info_path, 'w') as f:
        json.dump(training_info, f, indent=2)
    print(f"📄 Training info saved to: {info_path}")


def upload_model_to_wandb(config: Dict, model_dir: str, model_family: str, artifact_name: str = None, artifact_type: str = "model"):
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
        print("⏭️  W&B logging disabled, skipping artifact upload")
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
            artifact_name = f"{model_name}-{dataset_name}-lora"

        print(f"\n📤 Uploading model artifact to W&B...")
        print(f"   Artifact name: {artifact_name}")
        print(f"   Model directory: {model_dir}")

        # Create artifact
        artifact = wandb.Artifact(
            name=artifact_name,
            type=artifact_type,
            description=f"Fine-tuned {model_family or 'LLM'} model with LoRA adapters",
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
                "training_completed": datetime.now().isoformat()
            }
        )

        # Add model directory to artifact
        artifact.add_dir(model_dir)

        # Log artifact
        wandb.log_artifact(artifact)

        print(f"✅ Model artifact uploaded successfully!")
        print(f"   View at: {wandb.run.get_url()}/artifacts")

    except Exception as e:
        print(f"⚠️  Failed to upload model artifact to W&B: {e}")
        print("   Training will continue, but model was not uploaded")


def setup_wandb(config: Dict):
    """Setup Weights & Biases environment variables"""
    if 'wandb' in config:
        os.environ["WANDB_ENTITY"] = config['wandb'].get('entity', '')
        os.environ["WANDB_PROJECT"] = config['wandb'].get('project', 'SFT-Training')


def setup_chat_template(tokenizer: AutoTokenizer, model_name: str, config: Dict):
    """
    Setup chat template for the tokenizer if not present.

    Args:
        tokenizer: Tokenizer instance
        model_name: Model identifier
        config: Configuration dictionary
    """
    if hasattr(tokenizer, 'chat_template') and tokenizer.chat_template:
        print("✓ Tokenizer has built-in chat template")
        return

    print("⚠ Tokenizer missing chat template, attempting to set default...")

    # Check if user provided a custom template in config
    if 'chat_template' in config and 'template_string' in config['chat_template']:
        tokenizer.chat_template = config['chat_template']['template_string']
        print("✓ Applied custom chat template from config")
        return

    # Try to set a default based on model family
    family = detect_model_family(model_name)

    if family == 'llama':
        # Llama 3 style template with <|begin_of_text|>
        tokenizer.chat_template = (
            "{{ '<|begin_of_text|>' }}"
            "{% for message in messages %}"
            "{% if message['role'] == 'system' %}"
            "{{ '<|start_header_id|>system<|end_header_id|>\n\n' + message['content'] + '<|eot_id|>' }}"
            "{% elif message['role'] == 'user' %}"
            "{{ '<|start_header_id|>user<|end_header_id|>\n\n' + message['content'] + '<|eot_id|>' }}"
            "{% elif message['role'] == 'assistant' %}"
            "{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' + message['content'] + '<|eot_id|>' }}"
            "{% endif %}"
            "{% endfor %}"
            "{% if add_generation_prompt %}"
            "{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}"
            "{% endif %}"
        )
        print(f"✓ Applied default Llama 3 chat template with <|begin_of_text|>")

    elif family == 'qwen':
        # Qwen style template
        tokenizer.chat_template = (
            "{% for message in messages %}"
            "{{ '<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>\n' }}"
            "{% endfor %}"
            "{% if add_generation_prompt %}"
            "{{ '<|im_start|>assistant\n' }}"
            "{% endif %}"
        )
        print(f"✓ Applied default Qwen chat template")

    else:
        print(f"⚠ No default template for family '{family}'. Using generic template.")
        # Generic template
        tokenizer.chat_template = (
            "{% for message in messages %}"
            "{{ message['role'] + ': ' + message['content'] + '\n' }}"
            "{% endfor %}"
            "{% if add_generation_prompt %}"
            "{{ 'assistant: ' }}"
            "{% endif %}"
        )


def create_tokenizer(config: Dict) -> AutoTokenizer:
    """Create and configure tokenizer"""
    model_name = config['model']['name']

    # Detect model family for defaults
    family = detect_model_family(model_name)
    default_padding = MODEL_FAMILIES.get(family, {}).get('default_padding_side', 'right') if family else 'right'

    # Get padding side from config or use default
    padding_side = config['tokenizer'].get('padding_side', default_padding)

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=config['tokenizer'].get('trust_remote_code', True),
        padding_side=padding_side
    )

    # Ensure pad_token is set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        print(f"⚙ Set pad_token to eos_token: {tokenizer.pad_token}")

    # Setup chat template
    setup_chat_template(tokenizer, model_name, config)

    return tokenizer


def create_quantization_config(config: Dict) -> BitsAndBytesConfig:
    """Create BitsAndBytes quantization configuration"""
    quant_config = config['quantization']

    # Convert dtype string to torch dtype
    dtype_map = {
        'float16': torch.float16,
        'bfloat16': torch.bfloat16,
        'float32': torch.float32
    }
    compute_dtype = dtype_map.get(quant_config['bnb_4bit_compute_dtype'], torch.float16)

    return BitsAndBytesConfig(
        load_in_4bit=quant_config['load_in_4bit'],
        bnb_4bit_quant_type=quant_config['bnb_4bit_quant_type'],
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=quant_config['bnb_4bit_use_double_quant'],
    )


def create_model(config: Dict, bnb_config: BitsAndBytesConfig) -> AutoModelForCausalLM:
    """Create and configure model"""
    model = AutoModelForCausalLM.from_pretrained(
        config['model']['name'],
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=config['model'].get('trust_remote_code', True),
        torch_dtype=torch.float16,
    )

    # Prepare model for training
    model.config.use_cache = config['model'].get('use_cache', False)
    model = prepare_model_for_kbit_training(model)

    return model


def create_lora_config(config: Dict) -> LoraConfig:
    """Create LoRA configuration"""
    lora_cfg = config['lora']

    return LoraConfig(
        r=lora_cfg['r'],
        lora_alpha=lora_cfg['lora_alpha'],
        target_modules=lora_cfg['target_modules'],
        lora_dropout=lora_cfg['lora_dropout'],
        bias=lora_cfg['bias'],
        task_type=lora_cfg['task_type'],
    )


def print_trainable_parameters(model):
    """
    Print detailed information about trainable parameters.

    Args:
        model: The model to analyze
    """
    trainable_params = 0
    all_param = 0

    for name, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()

    trainable_percent = 100 * trainable_params / all_param

    print("\n" + "="*70)
    print("📊 TRAINABLE PARAMETERS SUMMARY")
    print("="*70)
    print(f"Total Parameters:        {all_param:,}")
    print(f"Trainable Parameters:    {trainable_params:,}")
    print(f"Non-Trainable Parameters: {all_param - trainable_params:,}")
    print(f"Trainable Percentage:    {trainable_percent:.4f}%")
    print("="*70)

    # Print breakdown by layer if model has print_trainable_parameters method
    if hasattr(model, 'print_trainable_parameters'):
        print("\n📋 Detailed Breakdown:")
        model.print_trainable_parameters()

    print()


# ============================================================================
# Dataset Processing Functions
# ============================================================================

def load_and_prepare_dataset(config: Dict) -> Tuple[Dataset, Dataset]:
    """Load and prepare dataset"""
    dataset_cfg = config['dataset']

    # Load dataset
    dataset = load_dataset(dataset_cfg['name'], split=dataset_cfg.get('split', 'train'))

    # Split train/eval
    dataset = dataset.train_test_split(
        test_size=dataset_cfg.get('test_size', 0.02),
        seed=dataset_cfg.get('seed', 42)
    )

    return dataset['train'], dataset['test']


def format_dataset_with_chat_template(
    example: Dict,
    tokenizer: AutoTokenizer,
    config: Dict
) -> Dict[str, str]:
    """
    Format dataset using tokenizer's chat template.
    Handles various dataset formats and converts to chat format.

    Args:
        example: Dataset example
        tokenizer: Tokenizer with chat template
        config: Configuration dictionary

    Returns:
        Dictionary with 'text' field containing formatted conversation
    """
    # Handle different dataset formats to extract messages
    messages = []

    # Try to extract conversation/messages from various fields
    if 'messages' in example:
        messages = example['messages']
    elif 'conversation' in example:
        messages = example['conversation']
    elif 'conversations' in example:
        messages = example['conversations']
    elif 'text' in example:
        # Already formatted text - return as is
        return {"text": example['text']}
    elif 'instruction' in example and 'output' in example:
        # Instruction-output format
        if config.get('chat_template', {}).get('use_system_message', False):
            messages.append({
                "role": "system",
                "content": config['chat_template'].get('system_message', '')
            })
        messages.append({"role": "user", "content": example['instruction']})
        messages.append({"role": "assistant", "content": example['output']})
    elif 'input' in example and 'output' in example:
        # Input-output format
        if config.get('chat_template', {}).get('use_system_message', False):
            messages.append({
                "role": "system",
                "content": config['chat_template'].get('system_message', '')
            })
        messages.append({"role": "user", "content": example['input']})
        messages.append({"role": "assistant", "content": example['output']})
    else:
        raise ValueError(f"Could not extract messages from example with keys: {example.keys()}")

    # Add system message if configured and not already present
    if config.get('chat_template', {}).get('use_system_message', False):
        if not messages or messages[0].get('role') != 'system':
            system_msg = {
                "role": "system",
                "content": config['chat_template'].get('system_message', '')
            }
            messages.insert(0, system_msg)

    # Normalize role names
    normalized_messages = []
    for msg in messages:
        if isinstance(msg, dict):
            role = msg.get("role", "user")
            content = msg.get("content", msg.get("value", ""))
        else:
            # Handle list format [role, content]
            role = msg[0] if len(msg) > 0 else "user"
            content = msg[1] if len(msg) > 1 else ""

        # Normalize role names
        if role in ["human", "Human"]:
            role = "user"
        elif role in ["ai", "bot", "gpt", "Assistant"]:
            role = "assistant"

        normalized_messages.append({"role": role, "content": content})

    # Use tokenizer's chat template to format
    try:
        text = tokenizer.apply_chat_template(
            normalized_messages,
            tokenize=False,
            add_generation_prompt=False
        )
        return {"text": text}
    except Exception as e:
        print(f"⚠ Error applying chat template: {e}")
        print(f"Messages: {normalized_messages}")
        raise


# ============================================================================
# Training Configuration
# ============================================================================

def create_training_config(config: Dict) -> SFTConfig:
    """Create SFT training configuration"""
    train_cfg = config['training']

    # Handle gradient checkpointing kwargs
    grad_ckpt_kwargs = train_cfg.get('gradient_checkpointing_kwargs', {})
    if isinstance(grad_ckpt_kwargs, dict) and 'use_reentrant' in grad_ckpt_kwargs:
        grad_ckpt_kwargs = {"use_reentrant": grad_ckpt_kwargs['use_reentrant']}
    else:
        grad_ckpt_kwargs = {"use_reentrant": False}

    return SFTConfig(
        output_dir=train_cfg['output_dir'],
        run_name=train_cfg.get('run_name', 'sft-training'),
        report_to=train_cfg.get('report_to', 'none'),

        # Batch configuration
        per_device_train_batch_size=train_cfg['per_device_train_batch_size'],
        per_device_eval_batch_size=train_cfg['per_device_eval_batch_size'],
        gradient_accumulation_steps=train_cfg['gradient_accumulation_steps'],

        # Memory optimization
        gradient_checkpointing=train_cfg.get('gradient_checkpointing', True),
        gradient_checkpointing_kwargs=grad_ckpt_kwargs,

        # Training parameters
        num_train_epochs=train_cfg['num_train_epochs'],
        learning_rate=train_cfg['learning_rate'],
        warmup_ratio=train_cfg.get('warmup_ratio', 0.03),

        # Optimizer
        optim=train_cfg.get('optim', 'paged_adamw_8bit'),
        weight_decay=train_cfg.get('weight_decay', 0.01),

        # Scheduler
        lr_scheduler_type=train_cfg.get('lr_scheduler_type', 'cosine'),

        # Precision
        bf16=train_cfg.get('bf16', False),
        fp16=train_cfg.get('fp16', True),

        # Logging & Evaluation
        logging_steps=train_cfg.get('logging_steps', 25),
        eval_strategy=train_cfg.get('eval_strategy', 'steps'),
        eval_steps=train_cfg.get('eval_steps', 200),
        save_strategy=train_cfg.get('save_strategy', 'steps'),
        save_steps=train_cfg.get('save_steps', 200),
        save_total_limit=train_cfg.get('save_total_limit', 2),

        # Data configuration
        max_seq_length=config['dataset']['max_length'],
        dataset_text_field=train_cfg.get('dataset_text_field', 'text'),
        packing=train_cfg.get('packing', False),
        dataloader_num_workers=train_cfg.get('dataloader_num_workers', 0),
        remove_unused_columns=train_cfg.get('remove_unused_columns', True),

        # Model selection
        load_best_model_at_end=train_cfg.get('load_best_model_at_end', True),
        metric_for_best_model=train_cfg.get('metric_for_best_model', 'eval_loss'),
        greater_is_better=train_cfg.get('greater_is_better', False),

        # Misc
        seed=train_cfg.get('seed', 42),
    )


def create_callbacks(config: Dict) -> List[TrainerCallback]:
    """Create training callbacks based on configuration"""
    callbacks = []
    callbacks_cfg = config.get('callbacks', {})

    # Early stopping
    if callbacks_cfg.get('early_stopping', {}).get('enabled', False):
        callbacks.append(
            EarlyStoppingCallback(
                early_stopping_patience=callbacks_cfg['early_stopping'].get('patience', 3),
                early_stopping_threshold=callbacks_cfg['early_stopping'].get('threshold', 0.0)
            )
        )

    # Custom logging
    if callbacks_cfg.get('custom_logging', {}).get('enabled', True):
        callbacks.append(CustomLoggingCallback())

    # Memory monitor
    if callbacks_cfg.get('memory_monitor', {}).get('enabled', True):
        callbacks.append(
            MemoryMonitorCallback(
                log_every_n_steps=callbacks_cfg['memory_monitor'].get('log_every_n_steps', 100)
            )
        )

    # Save best model
    if callbacks_cfg.get('save_best_model', {}).get('enabled', True):
        callbacks.append(SaveBestModelCallback())

    return callbacks


# ============================================================================
# Main Training Function
# ============================================================================

def main(config_path: str):
    """Main training function"""
    # Load configuration
    print(f"\n{'='*70}")
    print(f"📁 Loading configuration from: {config_path}")
    print(f"{'='*70}\n")
    config = load_config(config_path)

    # Get model info
    model_name = config['model']['name']
    model_family = detect_model_family(model_name)

    print(f"🤖 Model: {model_name}")
    if model_family:
        print(f"📋 Detected Model Family: {model_family}")
    else:
        print(f"⚠ Could not auto-detect model family. Using generic settings.")
    print()

    # Setup W&B
    setup_wandb(config)

    # Create tokenizer
    print("🔧 Creating tokenizer...")
    tokenizer = create_tokenizer(config)

    # Create quantization config
    print("⚙ Setting up 4-bit quantization...")
    bnb_config = create_quantization_config(config)

    # Create model
    print(f"📥 Loading model: {model_name}...")
    model = create_model(config, bnb_config)

    # Create LoRA config and apply
    print("🎯 Applying LoRA configuration...")
    lora_config = create_lora_config(config)
    model = get_peft_model(model, lora_config)

    # Print trainable parameters
    print_trainable_parameters(model)

    # Load and prepare dataset
    print(f"📊 Loading dataset: {config['dataset']['name']}...")
    train_dataset, eval_dataset = load_and_prepare_dataset(config)

    print(f"✓ Train samples: {len(train_dataset):,}")
    print(f"✓ Eval samples: {len(eval_dataset):,}")

    # Format datasets using chat template
    print("\n💬 Formatting datasets with chat template...")
    train_dataset = train_dataset.map(
        lambda x: format_dataset_with_chat_template(x, tokenizer, config),
        remove_columns=train_dataset.column_names,
        desc="Formatting train dataset"
    )
    eval_dataset = eval_dataset.map(
        lambda x: format_dataset_with_chat_template(x, tokenizer, config),
        remove_columns=eval_dataset.column_names,
        desc="Formatting eval dataset"
    )

    # Print sample
    print("\n📝 Sample formatted conversation:")
    print("-" * 70)
    print(train_dataset[0]['text'][:500] + "..." if len(train_dataset[0]['text']) > 500 else train_dataset[0]['text'])
    print("-" * 70)

    # Get response template for instruction masking
    response_template = get_response_template(model_name, tokenizer, config)

    # Create data collator for instruction masking
    print(f"\n🎭 Setting up instruction masking with response template: {repr(response_template)}")
    data_collator = DataCollatorForCompletionOnlyLM(
        response_template=response_template,
        tokenizer=tokenizer,
        mlm=False
    )

    # Create training configuration
    print("\n⚙ Creating training configuration...")
    training_args = create_training_config(config)

    # Create callbacks
    print("📞 Setting up callbacks...")
    callbacks = create_callbacks(config)

    # Clear cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        available_memory = torch.cuda.mem_get_info()[0] / 1024**3
        print(f"💾 Initial GPU Memory Available: {available_memory:.2f} GB")

    # Initialize trainer
    print("\n🚀 Initializing trainer...")
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,  # Add instruction masking
        processing_class=tokenizer,
        callbacks=callbacks,
    )

    # Print training summary
    print("\n" + "="*70)
    print("🎯 TRAINING CONFIGURATION SUMMARY")
    print("="*70)
    print(f"Model:                    {model_name}")
    print(f"Model Family:             {model_family or 'Unknown'}")
    print(f"LoRA Rank (r):            {config['lora']['r']}")
    print(f"LoRA Alpha:               {config['lora']['lora_alpha']}")
    print(f"Target Modules:           {', '.join(config['lora']['target_modules'][:3])}...")
    print(f"Batch Size:               {config['training']['per_device_train_batch_size']}")
    print(f"Gradient Accumulation:    {config['training']['gradient_accumulation_steps']}")
    print(f"Effective Batch Size:     {config['training']['per_device_train_batch_size'] * config['training']['gradient_accumulation_steps']}")
    print(f"Max Sequence Length:      {config['dataset']['max_length']}")
    print(f"Learning Rate:            {config['training']['learning_rate']}")
    print(f"Epochs:                   {config['training']['num_train_epochs']}")
    print(f"Optimizer:                {config['training'].get('optim', 'paged_adamw_8bit')}")
    print(f"Instruction Masking:      ✓ Enabled")
    print(f"Response Template:        {repr(response_template)}")
    if config.get('callbacks', {}).get('early_stopping', {}).get('enabled'):
        print(f"Early Stopping Patience:  {config['callbacks']['early_stopping']['patience']}")
    print("="*70 + "\n")

    # Train
    try:
        print("🏋️ Starting training...\n")
        trainer.train()
    except torch.cuda.OutOfMemoryError:
        print("\n❌ Out of Memory Error!")
        print("\n💡 Suggestions:")
        print("  • Reduce max_length in config (currently: {})".format(config['dataset']['max_length']))
        print("  • Reduce LoRA r value (currently: {})".format(config['lora']['r']))
        print("  • Increase gradient_accumulation_steps (currently: {})".format(config['training']['gradient_accumulation_steps']))
        print("  • Reduce batch size (currently: {})".format(config['training']['per_device_train_batch_size']))
        raise
    except Exception as e:
        print(f"\n❌ Training failed with error: {e}")
        raise

    # Save final model
    print("\n" + "="*70)
    print("💾 Training completed! Saving final model...")
    print("="*70 + "\n")

    final_model_dir = config['paths']['final_model_dir']

    # Save the LoRA adapter
    trainer.save_model(final_model_dir)

    # Save tokenizer
    tokenizer.save_pretrained(final_model_dir)

    # Save training info for later inference
    save_training_info(config, final_model_dir, model_family or 'unknown')

    # Upload model artifact to W&B
    upload_model_to_wandb(config, final_model_dir, model_family or 'unknown')

    # Also save the merged model if specified
    if config.get('save_merged_model', False):
        merged_dir = Path(final_model_dir).parent / f"{Path(final_model_dir).name}_merged"
        print(f"🔗 Saving merged model to: {merged_dir}")

        # Merge LoRA weights with base model
        merged_model = model.merge_and_unload()
        merged_model.save_pretrained(merged_dir)
        tokenizer.save_pretrained(merged_dir)

        # Upload merged model to W&B as separate artifact
        upload_model_to_wandb(
            config,
            str(merged_dir),
            model_family or 'unknown',
            artifact_name=f"{config['model']['name'].split('/')[-1]}-{config['dataset']['name'].split('/')[-1]}-merged",
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generic SFT Training Script - Supports Llama, Qwen, Mistral, and more"
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
