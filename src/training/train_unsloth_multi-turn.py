#multiturn training script
#Author: Tim 2 - DTP Finetuning

#import necessary libraries
import os
import sys
import json
import torch
import yaml
import argparse
from datetime import datetime
from unsloth import FastLanguageModel, is_bfloat16_supported
from unsloth.chat_templates import get_chat_template, train_on_responses_only
from datasets import load_dataset
from transformers import TrainingArguments, EarlyStoppingCallback, DataCollatorForSeq2Seq
from trl import SFTTrainer

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("W&B not available, logging will be disabled")

#yaml conf
def load_config(config_path: str) -> dict:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

#function setup wandb
def setup_wandb(config: dict, model_name: str):
    """Initialize Weights & Biases if available and configured."""
    if not WANDB_AVAILABLE:
        return False
    
    if config.get('training', {}).get('report_to') != 'wandb':
        return False
    
    if not os.environ.get("WANDB_API_KEY"):
        print("WANDB_API_KEY not found. Skipping W&B logging.")
        return False
    
    wandb_config = config.get('wandb', {})
    run_name = config['training'].get('run_name', f"SFT-{datetime.now().strftime('%Y%m%d-%H%M%S')}")
    
    wandb.init(
        entity=wandb_config.get('entity'),
        project=wandb_config.get('project', 'unsloth-training'),
        name=run_name,
        tags=wandb_config.get('tags', ['unsloth']),
        notes=wandb_config.get('notes', ''),
        config=config
    )
    print(f"[DONE] W&B initialized: {run_name}")
    return True

#function load model and tokenizer
def load_model_and_tokenizer(config: dict):
    model_name = config['model']['name']
    max_seq_length = config['dataset']['max_length']
    load_in_4bit = config.get('quantization', {}).get('load_in_4bit', True)
    
    print(f"Loading model: {model_name}")
    print(f"  Max sequence length: {max_seq_length}")
    print(f"  4-bit quantization: {load_in_4bit}")
    
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        dtype=None,  
        load_in_4bit=load_in_4bit,
        trust_remote_code=config.get('model', {}).get('trust_remote_code', True),
    )
    
    return model, tokenizer

#setup lora func
def setup_lora(model, config: dict):
    lora_config = config.get('lora', {})
    
    print(f"Configuring LoRA:")
    print(f"  r: {lora_config['r']}")
    print(f"  alpha: {lora_config['lora_alpha']}")
    print(f"  dropout: {lora_config['lora_dropout']}")
    print(f"  target_modules: {lora_config['target_modules']}")
    
    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_config['r'],
        lora_alpha=lora_config['lora_alpha'],
        lora_dropout=lora_config['lora_dropout'],
        target_modules=lora_config['target_modules'],
        use_gradient_checkpointing="unsloth",
        random_state=config['dataset'].get('seed', 42),
    )
    
    return model

#setup tokenizer func
def setup_tokenizer(tokenizer, config: dict):
    #paddingside qwen3
    padding_side = config.get('tokenizer', {}).get('padding_side', 'left')
    tokenizer.padding_side = padding_side
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    chat_template_name = config.get('chat_template', {}).get('name', 'qwen3')
    tokenizer = get_chat_template(tokenizer, chat_template=chat_template_name)
    
    print(f"[DONE] Tokenizer configured:")
    print(f"  Padding side: {padding_side}")
    print(f"  Chat template: {chat_template_name}")
    
    return tokenizer

#load and prepare dataset func
def prepare_dataset(config: dict, tokenizer):
    dataset_name = config['dataset']['name']
    split = config['dataset'].get('split', 'train')
    test_size = config['dataset'].get('test_size', 0.2)
    seed = config['dataset'].get('seed', 42)
    
    print(f"Loading dataset: {dataset_name}")
    
    dataset = load_dataset(dataset_name, split=split)
    print(f"  Total examples: {len(dataset)}")
    print(f"  Columns: {dataset.column_names}")
    
    sample = dataset[0]
    print(f"  Sample keys: {list(sample.keys())}")
    
    #convert messages to text func
    def convert_to_text(example):
        if 'messages' in example:
            messages = example['messages']
        elif 'conversations' in example:
            messages = []
            for turn in example['conversations']:
                role = turn.get('from', turn.get('role', 'user'))
                content = turn.get('value', turn.get('content', ''))
                if role in ['human', 'user']:
                    role = 'user'
                elif role in ['gpt', 'assistant', 'bot']:
                    role = 'assistant'
                messages.append({"role": role, "content": content})
        else:
            return {"text": str(example.get('text', ''))}
        
        #apply chat template
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            enable_thinking=False
        )
        return {"text": text}
    
    #check and convert
    if 'messages' in sample or 'conversations' in sample:
        print("  Converting messages to text format...")
        dataset = dataset.map(
            convert_to_text,
            remove_columns=dataset.column_names,
            desc="Formatting dataset"
        )
    elif 'text' not in sample:
        raise ValueError(f"Dataset must have 'messages', 'conversations', or 'text' column")
    
    #verify
    print(f"  Final columns: {dataset.column_names}")
    print(f"  Sample text (first 200 chars): {dataset[0]['text'][:200]}...")
    
    #split
    if test_size > 0:
        splits = dataset.train_test_split(test_size=test_size, seed=seed)
        train_dataset = splits['train']
        eval_dataset = splits['test']
        print(f"  Train size: {len(train_dataset)}")
        print(f"  Eval size: {len(eval_dataset)}")
    else:
        train_dataset = dataset
        eval_dataset = None
        print(f"  Train size: {len(train_dataset)} (no eval split)")
    
    return train_dataset, eval_dataset

#auto batch size detection func
def find_optimal_batch_size(model, tokenizer, train_dataset, config: dict, starting_batch_size: int = 32):
    print(f"\n{'='*60}")
    print("AUTO BATCH SIZE DETECTION")
    print(f"{'='*60}")
    
    if not torch.cuda.is_available():
        print("No CUDA device found, using batch_size=1")
        return 1
    
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # GB
    print(f"GPU Memory: {gpu_memory:.2f} GB")
    
    #conditional starting batch size
    if gpu_memory < 8:
        test_batch_size = 1
    elif gpu_memory < 16:
        test_batch_size = 2
    elif gpu_memory < 24:
        test_batch_size = 4
    else:
        test_batch_size = 8
    
    print(f"Starting test from batch_size={test_batch_size}")
    
    max_batch_size = test_batch_size
    sample_data = train_dataset.select(range(min(8, len(train_dataset))))
    
    def test_batch(batch_size):
        try:
            torch.cuda.empty_cache()
            
            sample_texts = [sample_data[i]['text'] for i in range(min(batch_size, len(sample_data)))]
            inputs = tokenizer(
                sample_texts,
                padding=True,
                truncation=True,
                max_length=config['dataset']['max_length'],
                return_tensors="pt"
            ).to(model.device)
            
            with torch.no_grad():
                _ = model(**inputs)
            
            torch.cuda.empty_cache()
            return True
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                torch.cuda.empty_cache()
                return False
            raise e
    
    #binarysearch
    while test_batch_size <= starting_batch_size:
        print(f"  Testing batch_size={test_batch_size}...", end=" ")
        if test_batch(test_batch_size):
            max_batch_size = test_batch_size
            print(f"[DONE] Success")
            test_batch_size *= 2
        else:
            print(f"[FAIL] OOM")
            break
    
    #foound optimal using intermediate values
    if max_batch_size < test_batch_size // 2:
        for bs in range(max_batch_size + 1, test_batch_size // 2):
            print(f"  Testing batch_size={bs}...", end=" ")
            if test_batch(bs):
                max_batch_size = bs
                print(f"[DONE] Success")
            else:
                print(f"[FAIL] OOM")
                break
    
    torch.cuda.empty_cache()
    
    print(f"\n[DONE] Optimal batch_size: {max_batch_size}")
    print(f"{'='*60}\n")
    
    return max_batch_size

#create trainer func
def create_trainer(model, tokenizer, train_dataset, eval_dataset, config: dict):
    train_config = config['training']
    advanced_config = config.get('advanced', {})
    
    #autodetect batch size if configure in yaml
    per_device_batch_size = train_config['per_device_train_batch_size']
    
    if per_device_batch_size == 'auto':
        print("Auto batch size detection enabled...")
        per_device_batch_size = find_optimal_batch_size(
            model, tokenizer, train_dataset, config,
            starting_batch_size=32
        )

        #update conf
        train_config['per_device_train_batch_size'] = per_device_batch_size
    elif isinstance(per_device_batch_size, str) and per_device_batch_size.startswith('auto:'):
        
        #ex: 'auto:16' to start testing from batch size 16
        starting_size = int(per_device_batch_size.split(':')[1])
        print(f"Auto batch size detection enabled (starting from {starting_size})...")
        per_device_batch_size = find_optimal_batch_size(
            model, tokenizer, train_dataset, config,
            starting_batch_size=starting_size
        )
        train_config['per_device_train_batch_size'] = per_device_batch_size
    
    model_dtype = next(model.parameters()).dtype
    use_bf16 = model_dtype == torch.bfloat16
    use_fp16 = model_dtype == torch.float16
    
    if not use_bf16 and not use_fp16:
        use_bf16 = is_bfloat16_supported()
        use_fp16 = not use_bf16
    
    print(f"Training precision: {'BF16' if use_bf16 else 'FP16'}")
    
    #advanced settings
    max_grad_norm = advanced_config.get('max_grad_norm', 1.0)
    use_neftune = advanced_config.get('use_neftune', False)
    neftune_alpha = advanced_config.get('neftune_noise_alpha', 5.0) if use_neftune else None
    
    #create training args
    training_args = TrainingArguments(
        output_dir=train_config['output_dir'],
        run_name=train_config.get('run_name', 'sft-training'),        
        per_device_train_batch_size=per_device_batch_size,
        per_device_eval_batch_size=train_config.get('per_device_eval_batch_size', per_device_batch_size),
        gradient_accumulation_steps=train_config['gradient_accumulation_steps'],        
        num_train_epochs=train_config['num_train_epochs'],
        learning_rate=train_config['learning_rate'],
        warmup_ratio=train_config.get('warmup_ratio', 0.05),
        weight_decay=train_config.get('weight_decay', 0.01),
        lr_scheduler_type=train_config.get('lr_scheduler_type', 'linear'),
        optim=train_config.get('optim', 'adamw_8bit'),
        fp16=use_fp16,
        bf16=use_bf16,
        max_grad_norm=max_grad_norm,
        logging_steps=train_config.get('logging_steps', 25),
        report_to=train_config.get('report_to', 'none'),
        eval_strategy=train_config.get('eval_strategy', 'steps') if eval_dataset else 'no',
        eval_steps=train_config.get('eval_steps', 50) if eval_dataset else None,
        save_strategy=train_config.get('save_strategy', 'steps'),
        save_steps=train_config.get('save_steps', 100),
        save_total_limit=train_config.get('save_total_limit', 3),
        load_best_model_at_end=train_config.get('load_best_model_at_end', True) if eval_dataset else False,
        metric_for_best_model=train_config.get('metric_for_best_model', 'eval_loss') if eval_dataset else None,
        greater_is_better=train_config.get('greater_is_better', False),
        neftune_noise_alpha=neftune_alpha,
        seed=config['dataset'].get('seed', 42),
        remove_unused_columns=True,
    )
    
    #early stopping
    callbacks = []
    if eval_dataset and train_config.get('early_stopping_patience'):
        callbacks.append(
            EarlyStoppingCallback(
                early_stopping_patience=train_config['early_stopping_patience']
            )
        )
        print(f"[DONE] Early stopping enabled (patience: {train_config['early_stopping_patience']})")
    
    # prevents the "excessive nesting" tensor creation error based on unsloth docs
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer)
    
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=training_args,
        dataset_text_field="text",  
        data_collator=data_collator,  
        max_seq_length=config['dataset']['max_length'],
        packing=train_config.get('packing', False),
        callbacks=callbacks if callbacks else None,
    )
    
    return trainer

#apply instruction masking func
def apply_instruction_masking(trainer, config: dict):
    chat_template = config.get('chat_template', {}).get('name', 'qwen3')
    
    if chat_template in ['qwen3', 'chatml', 'qwen-2.5']:
        instruction_part = "<|im_start|>user\n"
        response_part = "<|im_start|>assistant\n"
    elif chat_template == 'llama-3':
        instruction_part = "<|start_header_id|>user<|end_header_id|>\n\n"
        response_part = "<|start_header_id|>assistant<|end_header_id|>\n\n"
    else:
        instruction_part = "<|im_start|>user\n"
        response_part = "<|im_start|>assistant\n"
    
    print(f"Applying instruction masking:")
    print(f"  Instruction marker: {repr(instruction_part)}")
    print(f"  Response marker: {repr(response_part)}")
    
    trainer = train_on_responses_only(
        trainer,
        instruction_part=instruction_part,
        response_part=response_part,
    )
    
    return trainer

#save model func
def save_model(model, tokenizer, config: dict, training_loss: float, best_metric=None):
    final_dir = config['paths']['final_model_dir']
    
    print(f"Saving model to: {final_dir}")
    
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    training_info = {
        "model_name": config['model']['name'],
        "training_completed": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "dataset": config['dataset']['name'],
        "lora_config": config.get('lora', {}),
        "training_config": {
            "epochs": config['training']['num_train_epochs'],
            "learning_rate": config['training']['learning_rate'],
            "batch_size": config['training']['per_device_train_batch_size'],
            "gradient_accumulation": config['training']['gradient_accumulation_steps'],
        },
        "training_loss": training_loss,
        "best_eval_loss": best_metric,
        "chat_template": config.get('chat_template', {}).get('name', 'qwen3'),
    }
    
    with open(os.path.join(final_dir, "training_info.json"), 'w') as f:
        json.dump(training_info, f, indent=2)
    
    print(f"[DONE] Model and training info saved")


def upload_to_wandb(config: dict, final_dir: str, training_loss: float, best_metric=None):
    """Upload model to W&B as artifact."""
    if not WANDB_AVAILABLE or wandb.run is None:
        return
    
    try:
        artifact_name = config.get('wandb', {}).get('artifact_name', f"model-{wandb.run.name}")
        artifact = wandb.Artifact(
            name=artifact_name,
            type="model",
            description=f"Fine-tuned model from {wandb.run.name}",
            metadata={
                "model_name": config['model']['name'],
                "dataset": config['dataset']['name'],
                "training_loss": training_loss,
                "best_eval_loss": best_metric,
            }
        )
        artifact.add_dir(final_dir)
        wandb.log_artifact(artifact)
        print(f"[DONE] Model uploaded to W&B: {artifact_name}")
    except Exception as e:
        print(f"Warning: Failed to upload to W&B: {e}")


def main(config_path: str):
    """Main training function."""
    print("=" * 60)
    print("Unsloth Multi-Turn Training v2")
    print("=" * 60)
    
    # Load configurationzz
    config = load_config(config_path)
    print(f"[DONE] Config loaded: {config_path}")
    
    # Setup W&B
    wandb_enabled = setup_wandb(config, config['model']['name'])
    
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(config)
    print(f"[DONE] Model loaded")
    
    # Setup LoRA
    model = setup_lora(model, config)
    print(f"[DONE] LoRA configured")
    
    # Setup tokenizer with chat template
    tokenizer = setup_tokenizer(tokenizer, config)
    
    # Prepare dataset
    train_dataset, eval_dataset = prepare_dataset(config, tokenizer)
    print(f"[DONE] Dataset prepared")
    
    # Create trainer
    trainer = create_trainer(model, tokenizer, train_dataset, eval_dataset, config)
    print(f"[DONE] Trainer created")
    
    # Apply instruction masking
    trainer = apply_instruction_masking(trainer, config)
    print(f"[DONE] Instruction masking applied")
    
    # Print training summary
    print("\n" + "=" * 60)
    print("Training Summary")
    print("=" * 60)
    print(f"Model: {config['model']['name']}")
    print(f"Dataset: {config['dataset']['name']}")
    print(f"Train examples: {len(train_dataset)}")
    print(f"Eval examples: {len(eval_dataset) if eval_dataset else 'N/A'}")
    print(f"Batch size: {config['training']['per_device_train_batch_size']}")
    print(f"Gradient accumulation: {config['training']['gradient_accumulation_steps']}")
    effective_batch = config['training']['per_device_train_batch_size'] * config['training']['gradient_accumulation_steps']
    print(f"Effective batch size: {effective_batch}")
    print(f"Epochs: {config['training']['num_train_epochs']}")
    print(f"Learning rate: {config['training']['learning_rate']}")
    print(f"Output: {config['paths']['final_model_dir']}")
    print("=" * 60 + "\n")
    
    # Start training
    try:
        print("Starting training...")
        trainer_stats = trainer.train()
        training_loss = trainer_stats.training_loss
        best_metric = trainer.state.best_metric if hasattr(trainer.state, 'best_metric') else None
        
        print(f"\n[DONE] Training completed!")
        print(f"  Final loss: {training_loss:.4f}")
        if best_metric:
            print(f"  Best eval loss: {best_metric:.4f}")
        
    except torch.cuda.OutOfMemoryError:
        print("\n[FAIL] Out of Memory! Try reducing batch_size or max_length")
        sys.exit(1)
    except Exception as e:
        print(f"\n[FAIL] Training failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Save model
    save_model(model, tokenizer, config, training_loss, best_metric)
    
    # Upload to W&B
    if wandb_enabled:
        upload_to_wandb(config, config['paths']['final_model_dir'], training_loss, best_metric)
        wandb.finish()
        print("[DONE] W&B run finished")
    
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"Model saved to: {config['paths']['final_model_dir']}")
    print("\nNext steps:")
    print(f"  1. Run inference: python src/inference/gradio_inference.py --model {config['paths']['final_model_dir']}")
    print(f"  2. Push to hub: huggingface-cli upload {config['paths']['final_model_dir']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Unsloth Multi-Turn Training v2")
    parser.add_argument(
        "--config", "-c",
        type=str,
        required=True,
        help="Path to YAML configuration file"
    )
    args = parser.parse_args()
    
    if not os.path.isfile(args.config):
        print(f"Error: Config file not found: {args.config}")
        sys.exit(1)
    
    main(args.config)
