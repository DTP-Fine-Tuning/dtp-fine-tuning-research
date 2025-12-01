import os
import sys
from pathlib import Path
import torch
import yaml
import argparse
import json
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime

# unsloth
from unsloth import FastLanguageModel, is_bfloat16_supported
from unsloth.chat_templates import get_chat_template, train_on_responses_only

from datasets import load_dataset, Dataset
from trl import SFTTrainer, SFTConfig
import wandb

# prepare dataset
def load_and_prepare_dataset(config: Dict) -> Tuple[Dataset, Dataset]:
    dataset_name = config['dataset']['name']
    split = config['dataset'].get('split', 'train')
    test_size = config['dataset'].get('test_size', 0.1)
    seed = config['dataset'].get('seed', 42)

    print(f"Loading dataset: {dataset_name}")

    dataset = load_dataset(dataset_name, split=split)
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
    model = FastLanguageModel.from_pretrained(
        model_name,
        max_seq_len = config['dataset']['max_length'],
        dtype=None,
        load_in_4bit = config.get('quantization', {}).get('load_in_4bit', True),
        trust_remote_code = config.get('model', {}).get('trust_remote_code', False)
    )

    # config lora with unsloth
    lora_config = config.get('lora', {})
    model, tokenizer =  FastLanguageModel.get_peft_model(
        model,
        r = lora_config['r'],
        lora_alpha = lora_config['lora_alpha'],
        lora_dropout = lora_config['lora_dropout'],
        target_modules = lora_config['target_modules'],
        use_gradient_checkpointing = True,
        use_rslora = False,
        loftq_config = None
    )
    print(f"Maximum sequence length: {model.max_seq_len}")
    print(f"4-Bit quantization: {model.load_in_4bit}")
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
    train_dataset, eval_dataset = load_and_prepare_dataset(config)
    
    print(f"Training dataset size: {len(train_dataset)}")
    if eval_dataset:
        print(f"Evaluation dataset size: {len(eval_dataset)}")
    
    #print sample
    print("Sample formatted text")
    sample_txt = train_dataset[0]['text']
    print(sample_txt[:500] + "..." if len(sample_txt) > 500 else sample_txt)
    print("Dataset loading and preparation complete.")

    # create SFTConfig
    print("Creating SFT Configuration...")
    
    train_config = config['training']
    output_dir = train_config['output_dir']
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
        fp16= not is_bfloat16_supported(),

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
    print("SFT Configuration created.")

    # create SFT trainer with train_on_responses_only
    print("Initializing SFT Trainer...")
    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        args = sft_config,
        train_dataset = train_dataset,
        eval_dataset=  eval_dataset,
        max_seq_length = model.max_seq_len,
        dataset_num_proc = train_config.get('dataset_num_proc', 2),
    )

    # apply train_on_responses_only
    # Based on Qwen3 chat template: <|im_start|>role\ncontent<|im_end|>\n
    trainer = train_on_responses_only(
        trainer,
        instruction_part = "<|im_start|>user\n",
        response_part = "<|im_start|>assistant\n",
    )
    
    print("SFT Trainer initialized.")
    # training summary
    print("Training Configuration Summary:")
    print(f"Model: {model_name}")
    print(f"Dataset: {config['dataset']['name']}")
    print(f"Max Sequence Length: {model.max_seq_len}")
    print(f"LoRA Rank: {lora_config['r']}")
    print(f"LoRA Alpha: {lora_config['lora_alpha']}")
    print(f"LoRA Dropout: {lora_config['lora_dropout']}")
    print(f"Batch Size (per device): {train_config['per_device_train_batch_size']}")
    print(f"Gradient Accumulation Steps: {train_config['gradient_accumulation_steps']}")
    print(f"Effective Batch Size: {train_config['per_device_train_batch_size'] * train_config['gradient_accumulation_steps']}")
    print(f"Number of Epochs: {train_config['num_train_epochs']}")
    print(f"Learning Rate: {train_config['learning_rate']}")
    print(f"Warmup Ratio: {train_config['warmup_ratio']}")
    print(f"Optimizer: {train_config.get('optim', 'adamw_8bit')}")
    print(f"Epochs: {train_config['num_train_epochs']}")
    print(f"Output Directory: {output_dir}")

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

