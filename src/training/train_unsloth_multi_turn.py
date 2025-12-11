import os
import sys
import torch
import yaml
import argparse
from typing import Dict, Tuple
from datetime import datetime

# unsloth
from unsloth import FastLanguageModel, is_bfloat16_supported
from unsloth.chat_templates import get_chat_template, train_on_responses_only

from datasets import load_dataset, Dataset
from trl import SFTTrainer, SFTConfig
from transformers import EarlyStoppingCallback
import wandb

# prepare dataset
def load_and_prepare_dataset(config: Dict, tokenizer=None) -> Tuple[Dataset, Dataset]:
    dataset_name = config['dataset']['name']
    split = config['dataset'].get('split', 'train')
    test_size = config['dataset'].get('test_size', 0.1)
    seed = config['dataset'].get('seed', 42)

    print(f"Loading dataset: {dataset_name}")

    dataset = load_dataset(dataset_name, split=split)
    
    # Check dataset format and convert if needed
    sample = dataset[0]
    print(f"Dataset columns: {dataset.column_names}")
    
    # If dataset has 'messages' column, convert to 'text' using chat template
    if 'messages' in sample and 'text' not in sample and tokenizer is not None:
        print("Converting 'messages' format to 'text' using chat template...")
        def format_messages(example):
            text = tokenizer.apply_chat_template(
                example['messages'],
                tokenize=False,
                add_generation_prompt=False
            )
            return {"text": text}
        
        dataset = dataset.map(format_messages, num_proc=config.get('training', {}).get('dataset_num_proc', 2))
        print("Dataset converted to 'text' format.")
    
    if test_size > 0:
        dataset_split = dataset.train_test_split(test_size=test_size, seed=seed)
        train_dataset = dataset_split['train']
        test_dataset = dataset_split['test']
    else:
        train_dataset = dataset
        test_dataset = None
    
    return train_dataset, test_dataset

def setup_chat_template(tokenizer, config: Dict):
    tokenizer = get_chat_template(
        tokenizer,
        chat_template = config['chat_template']['name'])
    
    return tokenizer

def main(config_path: str):
    # load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    print(f"Loaded config from {config_path}")

    # detect model
    model_name = config['model']['name']
    print(f"Using model: {model_name}")

    # init wandb
    if config.get('training', {}).get('report_to') == 'wandb':
        if os.environ.get("WANDB_API_KEY"):
            wandb_config = config.get('wandb', {})
            wandb.init(
                entity = wandb_config.get('entity'),
                project = wandb_config.get('project'),
                name = config['training'].get('run_name', f"SFT-{model_name}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"),
                tags = wandb_config.get('tags', [] + ['unsloth']),
                notes = wandb_config.get('notes', ''),
                config=config
            )
            print("Initialized Weights & Biases logging.")
        else:
            print("WANDB_API_KEY not found in environment variables. Skipping Weights & Biases logging.")
            config['training']['report_to'] = None
    
    # load model n tokenizer
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name,
        max_seq_length = config['dataset']['max_length'],
        dtype = None,
        load_in_4bit = config.get('quantization', {}).get('load_in_4bit', True),
        trust_remote_code = config.get('model', {}).get('trust_remote_code', False)
    )

    # config lora with unsloth
    lora_config = config.get('lora', {})
    model = FastLanguageModel.get_peft_model(
        model,
        r = lora_config['r'],
        lora_alpha = lora_config['lora_alpha'],
        lora_dropout = lora_config['lora_dropout'],
        target_modules = lora_config['target_modules'],
        use_gradient_checkpointing = "unsloth",
        use_rslora = False,
        loftq_config = None
    )
    print(f"Maximum sequence length: {model.max_seq_length}")
    print(f"4-Bit quantization: {getattr(model, 'is_loaded_in_4bit', config.get('quantization', {}).get('load_in_4bit', True))}")
    print(f"LoRA configured with r={lora_config['r']}, alpha={lora_config['lora_alpha']}, dropout={lora_config['lora_dropout']}")
    print(f"target modules for LoRA: {lora_config['target_modules']}")
    print("Model and LoRA configuration complete.")

    # setup tokenizer and chat template
    print("Setting up tokenizer and chat template...")
    
    # set padding side for qwen3
    padding_side = config.get('tokenizer', {}).get('padding_side', 'left')
    tokenizer.padding_side = padding_side

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print(f"Tokenizer configured with padding side: {padding_side}, pad token: {tokenizer.pad_token}")

    tokenizer = setup_chat_template(tokenizer, config)
    print("Tokenizer and chat template setup complete.")

    # load dataset
    print("Loading and preparing dataset...")
    train_dataset, eval_dataset = load_and_prepare_dataset(config, tokenizer)
    
    print(f"Training dataset size: {len(train_dataset)}")
    if eval_dataset:
        print(f"Evaluation dataset size: {len(eval_dataset)}")
    
    #print sample
    print("Sample data from dataset:")
    sample = train_dataset[0]
    print(f"Available columns: {list(sample.keys())}")
    
    #Handle different dataset formats
    if 'text' in sample:
        sample_txt = sample['text']
        print(f"Text sample: {sample_txt[:500]}..." if len(sample_txt) > 500 else f"Text sample: {sample_txt}")
    elif 'messages' in sample:
        print(f"Messages sample: {sample['messages'][:2]}...")
    elif 'conversations' in sample:
        print(f"Conversations sample: {sample['conversations'][:2]}...")
    else:
        print(f"First sample keys: {list(sample.keys())}")
        for key in list(sample.keys())[:3]:
            val = str(sample[key])
            print(f"  {key}: {val[:200]}..." if len(val) > 200 else f"  {key}: {val}")
    
    print("Dataset loading and preparation complete.")

    #create SFTConfig
    print("Creating SFT Configuration...")
    
    train_config = config['training']
    advanced_config = config.get('advanced', {})
    output_dir = train_config['output_dir']
    
    # Get advanced settings with defaults
    max_grad_norm = advanced_config.get('max_grad_norm', 1.0)
    use_neftune = advanced_config.get('use_neftune', False)
    neftune_noise_alpha = advanced_config.get('neftune_noise_alpha', 5.0) if use_neftune else None
    packing = train_config.get('packing', False)
    
    # Precision selection: FP16 vs BF16
    # Based on "Defeating the Training-Inference Mismatch via FP16" paper:
    # - FP16 has higher precision (11-bit mantissa) and better training-inference consistency
    # - BF16 has lower precision (8-bit mantissa) but larger dynamic range
    # If explicitly set in config, use those values; otherwise auto-detect
    precision_config = config.get('precision', {})
    
    if 'fp16' in precision_config or 'bf16' in precision_config:
        use_fp16 = precision_config.get('fp16', False)
        use_bf16 = precision_config.get('bf16', False)
        
        if use_fp16 and use_bf16:
            print("WARNING: Both fp16 and bf16 set to True. Using fp16 for better training-inference consistency.")
            use_bf16 = False
        elif not use_fp16 and not use_bf16:
            use_bf16 = is_bfloat16_supported()
            use_fp16 = not use_bf16
            print(f"NOTE: No precision specified, auto-detected: {'bf16' if use_bf16 else 'fp16'}")
        
        precision_source = "config (explicit)"
    else:
        use_bf16 = is_bfloat16_supported()
        use_fp16 = not use_bf16
        precision_source = "auto-detected"
    
    print(f"Precision: {'FP16' if use_fp16 else 'BF16'} ({precision_source})")
    if use_fp16:
        print("FP16 provides better training-inference consistency (11-bit mantissa)")
    else:
        print("BF16 provides larger dynamic range but lower precision (8-bit mantissa)")
    
    sft_config = SFTConfig(
        output_dir = output_dir,
        run_name = train_config.get('run_name', f"SFT-{model_name}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"),
        per_device_train_batch_size = train_config['per_device_train_batch_size'],
        per_device_eval_batch_size = train_config['per_device_eval_batch_size'],
        gradient_accumulation_steps = train_config['gradient_accumulation_steps'],
        num_train_epochs = train_config['num_train_epochs'],
        learning_rate= train_config['learning_rate'],
        warmup_ratio= train_config['warmup_ratio'],
        optim = train_config.get('optim', 'adamw_8bit'),
        weight_decay= train_config['weight_decay'],
        lr_scheduler_type= train_config['lr_scheduler_type'],
        fp16= use_fp16,
        bf16= use_bf16,
        max_grad_norm= max_grad_norm,
        neftune_noise_alpha= neftune_noise_alpha,
        packing= packing,

        # logging & eval
        logging_steps= train_config['logging_steps'],
        eval_strategy= train_config.get('eval_strategy', train_config.get('evaluation_strategy', 'steps')),
        eval_steps= train_config['eval_steps'],
        save_strategy= train_config['save_strategy'],
        save_steps= train_config['save_steps'],
        save_total_limit= train_config['save_total_limit'],

        # model selection
        load_best_model_at_end= train_config['load_best_model_at_end'],
        metric_for_best_model= train_config.get('metric_for_best_model', 'eval_loss'),
        greater_is_better= train_config.get('greater_is_better', False),

        seed= config['dataset'].get('seed', 42),
        report_to= train_config.get('report_to', None),
    )
    
    # Log advanced config
    print(f"Advanced settings applied:")
    print(f"  - max_grad_norm: {max_grad_norm}")
    print(f"  - NEFTune: {'enabled (alpha={})'.format(neftune_noise_alpha) if use_neftune else 'disabled'}")
    print(f"  - Packing: {'enabled' if packing else 'disabled'}")
    print(f"  - bf16: {is_bfloat16_supported()}, fp16: {not is_bfloat16_supported()}")
    print("SFT Configuration created.")

    # create SFT trainer with train_on_responses_only
    print("Initializing SFT Trainer...")
    
    # Setup callbacks
    callbacks = []
    early_stopping_patience = train_config.get('early_stopping_patience', None)
    if early_stopping_patience:
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=early_stopping_patience))
        print(f"Early stopping enabled with patience: {early_stopping_patience}")
    
    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        args = sft_config,
        train_dataset = train_dataset,
        eval_dataset=  eval_dataset,
        max_seq_length = model.max_seq_length,
        callbacks = callbacks if callbacks else None,
        dataset_num_proc = train_config.get('dataset_num_proc', 2),
    )

    # apply train_on_responses_only
    # Based on Qwen3 chat template: <|im_start|>role\ncontent<|im_end|>\n
    # WARNING: Packing can cause "bleeding" between conversations in multi-turn training
    # When packing is enabled, multiple conversations are concatenated without clear boundaries
    # This can confuse the model as assistant responses may bleed into the next conversation
    if packing:
        print("\n" + "!"*60)
        print("WARNING: Packing is ENABLED with train_on_responses_only!")
        print("This may cause 'bleeding' between conversations in multi-turn training.")
        print("Recommendation: Set 'packing: false' in config for multi-turn datasets.")
        print("!"*60 + "\n")
    
    trainer = train_on_responses_only(
        trainer,
        instruction_part = "<|im_start|>user\n",
        response_part = "<|im_start|>assistant\n",
    )
    
    print("SFT Trainer initialized.")
    # training summary
    print("\n" + "="*60)
    print("Training Configuration Summary:")
    print("="*60)
    print(f"Model: {model_name}")
    print(f"Dataset: {config['dataset']['name']}")
    print(f"Max Sequence Length: {model.max_seq_length}")
    print(f"\n[LoRA Configuration]")
    print(f"  Rank: {lora_config['r']}")
    print(f"  Alpha: {lora_config['lora_alpha']}")
    print(f"  Dropout: {lora_config['lora_dropout']}")
    print(f"  Target Modules: {lora_config['target_modules']}")
    print(f"\n[Training Parameters]")
    print(f"  Batch Size (per device): {train_config['per_device_train_batch_size']}")
    print(f"  Gradient Accumulation Steps: {train_config['gradient_accumulation_steps']}")
    print(f"  Effective Batch Size: {train_config['per_device_train_batch_size'] * train_config['gradient_accumulation_steps']}")
    print(f"  Number of Epochs: {train_config['num_train_epochs']}")
    print(f"  Learning Rate: {train_config['learning_rate']}")
    print(f"  Warmup Ratio: {train_config['warmup_ratio']}")
    print(f"  Optimizer: {train_config.get('optim', 'adamw_8bit')}")
    print(f"  LR Scheduler: {train_config['lr_scheduler_type']}")
    print(f"\n[Advanced Settings]")
    print(f"  Max Grad Norm: {max_grad_norm}")
    print(f"  NEFTune: {'Enabled (alpha={})'.format(neftune_noise_alpha) if use_neftune else 'Disabled'}")
    print(f"  Packing: {'Enabled' if packing else 'Disabled'}")
    print(f"  Precision: {'FP16' if use_fp16 else 'BF16'} ({precision_source})")
    print(f"  Flash Attention: Enabled (via Unsloth)")
    print(f"\n[Output]")
    print(f"  Output Directory: {output_dir}")
    print("="*60 + "\n")

    # start training
    try:
        print("Starting training...")
        trainer_stats = trainer.train()
        print("Training completed successfully.")
        print(f"Training loss: {trainer_stats.training_loss:.4f}")
    except torch.cuda.OutOfMemoryError:
        print("ERROR: Out of Memory during training. Try reducing the batch size or gradient accumulation steps.")
        sys.exit(1)
    except Exception as e:
        print(f"ERROR: An unexpected error occurred during training: {e}")
        sys.exit(1)

    # save model and tokenizer
    print("Saving fine-tuned model and tokenizer...")
    final_model_dir = config['paths']['final_model_dir']

    model.save_pretrained(final_model_dir)
    tokenizer.save_pretrained(final_model_dir)
    print(f"Model and tokenizer saved to {final_model_dir}")

    # Save training_info.json for inference
    import json
    training_info = {
        "model_name": model_name,
        "training_completed": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "dataset_info": {
            "name": config['dataset']['name'],
            "max_length": config['dataset']['max_length'],
        },
        "lora_config": {
            "r": lora_config['r'],
            "lora_alpha": lora_config['lora_alpha'],
            "lora_dropout": lora_config['lora_dropout'],
            "target_modules": lora_config['target_modules'],
        },
        "training_config": {
            "epochs": train_config['num_train_epochs'],
            "learning_rate": train_config['learning_rate'],
            "batch_size": train_config['per_device_train_batch_size'],
            "gradient_accumulation_steps": train_config['gradient_accumulation_steps'],
            "optimizer": train_config.get('optim', 'adamw_8bit'),
        },
        "advanced_config": {
            "max_grad_norm": max_grad_norm,
            "neftune_enabled": use_neftune,
            "neftune_noise_alpha": neftune_noise_alpha,
            "packing": packing,
            "precision": "fp16" if use_fp16 else "bf16",
            "precision_source": precision_source,
        },
        "chat_template": {
            "name": config['chat_template']['name'],
            "system_message": "You are a helpful assistant.",
            "use_system_message": True,
        },
        "training_loss": trainer_stats.training_loss,
        "best_eval_loss": trainer.state.best_metric if trainer.state.best_metric else None,
    }
    
    training_info_path = os.path.join(final_model_dir, "training_info.json")
    with open(training_info_path, 'w') as f:
        json.dump(training_info, f, indent=2)
    print(f"Training info saved to {training_info_path}")

    # print training summary
    if trainer.state.best_metric is not None:
        print(f"Best model metric ({sft_config.metric_for_best_model}): {trainer.state.best_metric:.4f} at step {trainer.state.best_model_checkpoint}")
    print("Training process completed.")

    # upload model to wandb as artifact
    if wandb.run is not None:
        print("Uploading model to Weights & Biases...")
        try:
            artifact_name = config.get('wandb', {}).get('artifact_name', f"model-{wandb.run.name}")
            artifact = wandb.Artifact(
                name=artifact_name,
                type="model",
                description=f"Fine-tuned model from run {wandb.run.name}",
                metadata={
                    "model_name": model_name,
                    "dataset": config['dataset']['name'],
                    "lora_r": lora_config['r'],
                    "lora_alpha": lora_config['lora_alpha'],
                    "epochs": train_config['num_train_epochs'],
                    "learning_rate": train_config['learning_rate'],
                    "training_loss": trainer_stats.training_loss,
                    "best_eval_loss": trainer.state.best_metric if trainer.state.best_metric else None,
                }
            )
            artifact.add_dir(final_model_dir)
            wandb.log_artifact(artifact)
            print(f"Model uploaded to W&B as artifact: {artifact_name}")
        except Exception as e:
            print(f"WARNING: Failed to upload model to W&B: {e}")

    #close wandb
    if wandb.run is not None:
        wandb.finish()
        print("Weights & Biases run finished.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a language model using Unsloth and TRL SFTTrainer.")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the YAML configuration file."
    )
    args = parser.parse_args()

    if not os.path.isfile(args.config):
        print(f"ERROR: Config file {args.config} does not exist.")
        sys.exit(1)
    
    main(args.config)

