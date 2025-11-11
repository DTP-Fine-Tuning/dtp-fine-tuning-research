import os
import sys
import torch
import yaml
import argparse
import json
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
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel
import wandb
from datetime import datetime


# Custom Callbacks
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
                print(f"\nNew best model! Eval Loss: {current_metric:.4f}")


def load_config(config_path: str) -> Dict:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def save_training_info(config: Dict, output_dir: str):
    """Save training configuration and metadata for later inference"""
    training_info = {
        "model_name": config['model']['name'],
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
    print(f"Training info saved to: {info_path}")


def setup_wandb(config: Dict):
    """Setup Weights & Biases environment variables"""
    if 'wandb' in config:
        os.environ["WANDB_ENTITY"] = config['wandb']['entity']
        os.environ["WANDB_PROJECT"] = config['wandb']['project']


def create_tokenizer(config: Dict) -> AutoTokenizer:
    """Create and configure tokenizer for Qwen2"""
    tokenizer_config = config.get('tokenizer', {})
    
    tokenizer = AutoTokenizer.from_pretrained(
        config['model']['name'],
        trust_remote_code=tokenizer_config.get('trust_remote_code', True),
        padding_side=tokenizer_config.get('padding_side', 'left')
    )
    
    # Qwen2 models use eos_token as pad_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
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
    """Create and configure Qwen2 model"""
    model = AutoModelForCausalLM.from_pretrained(
        config['model']['name'],
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=config['model']['trust_remote_code'],
        torch_dtype=torch.float16,
    )
    
    # Prepare model for training
    model.config.use_cache = config['model']['use_cache']
    model = prepare_model_for_kbit_training(model)
    
    return model


def create_lora_config(config: Dict) -> LoraConfig:
    """Create LoRA configuration for Qwen2"""
    lora_cfg = config['lora']
    
    return LoraConfig(
        r=lora_cfg['r'],
        lora_alpha=lora_cfg['lora_alpha'],
        target_modules=lora_cfg['target_modules'],
        lora_dropout=lora_cfg['lora_dropout'],
        bias=lora_cfg['bias'],
        task_type=lora_cfg['task_type'],
    )


def load_and_prepare_dataset(config: Dict) -> Tuple[Dataset, Dataset]:
    """Load and prepare dataset"""
    dataset_cfg = config['dataset']
    
    # Load dataset
    dataset = load_dataset(dataset_cfg['name'], split=dataset_cfg['split'])
    
    # Split train/eval
    dataset = dataset.train_test_split(
        test_size=dataset_cfg['test_size'],
        seed=dataset_cfg['seed']
    )
    
    return dataset['train'], dataset['test']


def format_messages_to_qwen2(messages: List[Dict], config: Dict) -> str:
    """
    Format messages array to Qwen2 chat template.
    Qwen2 uses ChatML format:
    <|im_start|>system
    {system_message}<|im_end|>
    <|im_start|>user
    {user_message}<|im_end|>
    <|im_start|>assistant
    {assistant_message}<|im_end|>
    """
    formatted_text = ""
    
    for message in messages:
        role = message.get("role", "")
        content = message.get("content", "")
        
        if role and content:
            formatted_text += f"<|im_start|>{role}\n{content}<|im_end|>\n"
    
    return formatted_text


def format_dataset(example: Dict, tokenizer: AutoTokenizer, config: Dict) -> Dict[str, str]:
    """
    Format dataset example to Qwen2 chat format.
    Handles datasets with 'messages' field containing array of message objects.
    """
    # Get messages from the example
    messages = example.get('messages', [])
    
    # Use tokenizer's apply_chat_template if available
    if hasattr(tokenizer, 'apply_chat_template') and tokenizer.chat_template:
        try:
            text = tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=False
            )
            return {"text": text}
        except Exception as e:
            print(f"Error applying chat template: {e}, falling back to manual formatting")
    
    # Fallback to manual formatting
    text = format_messages_to_qwen2(messages, config)
    return {"text": text}


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
        run_name=train_cfg['run_name'],
        report_to=train_cfg['report_to'],
        
        # Batch configuration
        per_device_train_batch_size=train_cfg['per_device_train_batch_size'],
        per_device_eval_batch_size=train_cfg['per_device_eval_batch_size'],
        gradient_accumulation_steps=train_cfg['gradient_accumulation_steps'],
        
        # Memory optimization
        gradient_checkpointing=train_cfg['gradient_checkpointing'],
        gradient_checkpointing_kwargs=grad_ckpt_kwargs,
        
        # Training parameters
        num_train_epochs=train_cfg['num_train_epochs'],
        learning_rate=train_cfg['learning_rate'],
        warmup_ratio=train_cfg['warmup_ratio'],
        
        # Optimizer
        optim=train_cfg['optim'],
        weight_decay=train_cfg['weight_decay'],
        
        # Scheduler
        lr_scheduler_type=train_cfg['lr_scheduler_type'],
        
        # Precision
        bf16=train_cfg['bf16'],
        fp16=train_cfg['fp16'],
        
        # Logging & Evaluation
        logging_steps=train_cfg['logging_steps'],
        eval_strategy=train_cfg['eval_strategy'],
        eval_steps=train_cfg['eval_steps'],
        save_strategy=train_cfg['save_strategy'],
        save_steps=train_cfg['save_steps'],
        save_total_limit=train_cfg['save_total_limit'],
        
        # Data configuration
        max_seq_length=config['dataset']['max_length'],
        dataset_text_field=train_cfg['dataset_text_field'],
        packing=train_cfg['packing'],
        dataloader_num_workers=train_cfg['dataloader_num_workers'],
        remove_unused_columns=train_cfg['remove_unused_columns'],
        
        # Model selection
        load_best_model_at_end=train_cfg['load_best_model_at_end'],
        metric_for_best_model=train_cfg['metric_for_best_model'],
        greater_is_better=train_cfg['greater_is_better'],
        
        # Misc
        seed=train_cfg['seed'],
    )


def create_callbacks(config: Dict) -> List[TrainerCallback]:
    """Create training callbacks based on configuration"""
    callbacks = []
    callbacks_cfg = config['callbacks']
    
    # Early stopping
    if callbacks_cfg['early_stopping']['enabled']:
        callbacks.append(
            EarlyStoppingCallback(
                early_stopping_patience=callbacks_cfg['early_stopping']['patience'],
                early_stopping_threshold=callbacks_cfg['early_stopping']['threshold']
            )
        )
    
    # Custom logging
    if callbacks_cfg['custom_logging']['enabled']:
        callbacks.append(CustomLoggingCallback())
    
    # Memory monitor
    if callbacks_cfg['memory_monitor']['enabled']:
        callbacks.append(
            MemoryMonitorCallback(
                log_every_n_steps=callbacks_cfg['memory_monitor']['log_every_n_steps']
            )
        )
    
    # Save best model
    if callbacks_cfg['save_best_model']['enabled']:
        callbacks.append(SaveBestModelCallback())
    
    return callbacks


def main(config_path: str):
    """Main training function"""
    # Load configuration
    print(f"\n{'='*50}")
    print(f"Loading configuration from: {config_path}")
    print(f"{'='*50}\n")
    config = load_config(config_path)
    
    # Setup W&B
    setup_wandb(config)
    
    # Create tokenizer
    print("Creating tokenizer for Qwen2...")
    tokenizer = create_tokenizer(config)
    
    # Create quantization config
    print("Setting up 4-bit quantization...")
    bnb_config = create_quantization_config(config)
    
    # Create model
    print(f"Loading model: {config['model']['name']}...")
    model = create_model(config, bnb_config)
    
    # Create LoRA config and apply
    print("Applying LoRA configuration...")
    lora_config = create_lora_config(config)
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # Load and prepare dataset
    print(f"\nLoading dataset: {config['dataset']['name']}...")
    train_dataset, eval_dataset = load_and_prepare_dataset(config)
    
    # Format datasets
    print("Formatting datasets with Qwen2 chat template...")
    train_dataset = train_dataset.map(
        lambda x: format_dataset(x, tokenizer, config),
        remove_columns=train_dataset.column_names
    )
    eval_dataset = eval_dataset.map(
        lambda x: format_dataset(x, tokenizer, config),
        remove_columns=eval_dataset.column_names
    )
    
    # Print sample
    print("\nSample formatted text:")
    print("-" * 50)
    print(train_dataset[0]['text'][:500])
    print("-" * 50)
    
    # Create training configuration
    print("Creating training configuration...")
    training_args = create_training_config(config)
    
    # Create callbacks
    print("Setting up callbacks...")
    callbacks = create_callbacks(config)
    
    # Clear cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        available_memory = torch.cuda.mem_get_info()[0] / 1024**3
        print(f"Initial GPU Memory Available: {available_memory:.2f} GB")
    
    # Initialize trainer
    print("\nInitializing trainer...")
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        callbacks=callbacks,
    )
    
    # Train
    print("\n" + "="*50)
    print("Starting training with configuration:")
    print(f"  - Model: {config['model']['name']}")
    print(f"  - LoRA r: {config['lora']['r']}")
    print(f"  - Batch size: {config['training']['per_device_train_batch_size']}")
    print(f"  - Gradient accumulation: {config['training']['gradient_accumulation_steps']}")
    print(f"  - Max length: {config['dataset']['max_length']}")
    print(f"  - Epochs: {config['training']['num_train_epochs']}")
    print("="*50 + "\n")
    
    try:
        trainer.train()
    except torch.cuda.OutOfMemoryError:
        print("\nOut of memory error!")
        print("Suggestions:")
        print("  - Reduce max_length in config")
        print("  - Reduce LoRA r value")
        print("  - Increase gradient_accumulation_steps")
        raise
    except Exception as e:
        print(f"\nTraining failed with error: {e}")
        raise
    
    # Save final model
    print("\n" + "="*50)
    print("Training completed! Saving final model...")
    print("="*50 + "\n")
    
    final_model_dir = config['paths']['final_model_dir']
    
    # Save the LoRA adapter
    trainer.save_model(final_model_dir)
    
    # Save tokenizer
    tokenizer.save_pretrained(final_model_dir)
    
    # Save training info for later inference
    save_training_info(config, final_model_dir)
    
    # Also save the merged model if specified
    if config.get('save_merged_model', False):
        merged_dir = Path(final_model_dir).parent / f"{Path(final_model_dir).name}_merged"
        print(f"Saving merged model to: {merged_dir}")
        
        # Merge LoRA weights with base model
        merged_model = model.merge_and_unload()
        merged_model.save_pretrained(merged_dir)
        tokenizer.save_pretrained(merged_dir)
    
    # Print training summary
    if trainer.state.best_metric is not None:
        print(f"\nBest eval loss achieved: {trainer.state.best_metric:.4f}")
        if trainer.state.best_model_checkpoint:
            print(f"Best model checkpoint: {trainer.state.best_model_checkpoint}")
    
    print("\nTraining pipeline completed successfully!")
    print(f"Model saved to: {final_model_dir}")
    
    # Final memory check
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"\nFinal GPU Memory - Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Qwen2 model with SFT using config file")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the YAML configuration file"
    )
    
    args = parser.parse_args()
    
    # Verify config file exists
    if not Path(args.config).exists():
        print(f"Error: Configuration file not found at {args.config}")
        sys.exit(1)
    
    main(args.config)
