#single training script
#Author: Tim 2 - DTP Finetuning

# import neccesary libraries
import os
import sys
import json
import torch
import yaml
import argparse
from typing import Dict, Tuple, Optional
from datetime import datetime
from unsloth import FastLanguageModel, is_bfloat16_supported
from unsloth.chat_templates import get_chat_template, train_on_responses_only
from datasets import load_dataset, Dataset
from trl import SFTTrainer, SFTConfig
from transformers import EarlyStoppingCallback
import wandb

#format alpaca style func
def format_alpaca_style(example: Dict, tokenizer) -> Dict:
    instruction = example.get('instruction', '')
    input_text = example.get('input', '')
    output = example.get('output', example.get('response', ''))
    
    if input_text:
        user_content = f"{instruction}\n\n{input_text}"
    else:
        user_content = instruction
    
    messages = [
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": output}
    ]
    
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False
    )
    
    return {"text": text}

#format prompt completion func
def format_prompt_completion(example: Dict, tokenizer) -> Dict:
    prompt = example.get('prompt', example.get('question', ''))
    completion = example.get('completion', example.get('answer', example.get('response', '')))
    
    messages = [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": completion}
    ]
    
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False
    )
    
    return {"text": text}

#format qa style func
def format_qa_style(example: Dict, tokenizer) -> Dict:
    question = example.get('question', example.get('query', ''))
    answer = example.get('answer', example.get('response', ''))
    
    messages = [
        {"role": "user", "content": question},
        {"role": "assistant", "content": answer}
    ]
    
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False
    )
    
    return {"text": text}

#format input output func
def format_input_output(example: Dict, tokenizer) -> Dict:
    input_text = example.get('input', example.get('text', ''))
    output_text = example.get('output', example.get('label', example.get('target', '')))
    
    messages = [
        {"role": "user", "content": input_text},
        {"role": "assistant", "content": output_text}
    ]
    
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False
    )
    
    return {"text": text}

#auto detect and format dataset func
def detect_and_format_dataset(dataset: Dataset, tokenizer, config: Dict) -> Dataset:
    sample = dataset[0]
    columns = set(sample.keys())
    
    print(f"Detected columns: {columns}")
    
    if 'text' in columns:
        print("Dataset already has 'text' column, using as-is.")
        return dataset
    
    if 'messages' in columns:
        print("Detected 'messages' format, converting to text...")
        
        def format_messages(example):
            messages = example['messages']
            formatted_messages = []
            
            for msg in messages:
                role = msg.get('role', '')
                content = msg.get('content', '')
                
                if isinstance(content, (list, dict)):
                    content = json.dumps(content, ensure_ascii=False, indent=2)
                
                formatted_messages.append({
                    "role": role,
                    "content": str(content)
                })
            
            # For single-turn, take first user and first assistant (skip system if exists)
            single_turn_messages = []
            user_found = False
            
            for msg in formatted_messages:
                if msg['role'] == 'system':
                    single_turn_messages.append(msg)
                elif msg['role'] == 'user' and not user_found:
                    single_turn_messages.append(msg)
                    user_found = True
                elif msg['role'] == 'assistant' and user_found:
                    single_turn_messages.append(msg)
                    break
            
            if len(single_turn_messages) >= 2:
                text = tokenizer.apply_chat_template(
                    single_turn_messages,
                    tokenize=False,
                    add_generation_prompt=False
                )
                return {"text": text}
            
            text = tokenizer.apply_chat_template(
                formatted_messages,
                tokenize=False,
                add_generation_prompt=False
            )
            return {"text": text}
        
        return dataset.map(
            format_messages,
            num_proc=config.get('training', {}).get('dataset_num_proc', 2)
        )
    
    if 'instruction' in columns and ('output' in columns or 'response' in columns):
        print("Detected Alpaca-style format (instruction, input, output)")
        return dataset.map(
            lambda x: format_alpaca_style(x, tokenizer),
            num_proc=config.get('training', {}).get('dataset_num_proc', 2)
        )
    
    if 'prompt' in columns and ('completion' in columns or 'response' in columns):
        print("Detected prompt-completion format")
        return dataset.map(
            lambda x: format_prompt_completion(x, tokenizer),
            num_proc=config.get('training', {}).get('dataset_num_proc', 2)
        )
    
    if ('question' in columns or 'query' in columns) and ('answer' in columns or 'response' in columns):
        print("Detected Q&A format")
        return dataset.map(
            lambda x: format_qa_style(x, tokenizer),
            num_proc=config.get('training', {}).get('dataset_num_proc', 2)
        )
    
    if 'input' in columns and 'output' in columns:
        print("Detected simple input-output format")
        return dataset.map(
            lambda x: format_input_output(x, tokenizer),
            num_proc=config.get('training', {}).get('dataset_num_proc', 2)
        )
    
    raise ValueError(
        f"Could not detect dataset format. Found columns: {columns}\n"
        "Supported formats:\n"
        "  - Alpaca: instruction, input (optional), output\n"
        "  - Prompt-Completion: prompt, completion\n"
        "  - Q&A: question, answer\n"
        "  - Simple: input, output\n"
        "  - Messages: messages (list of role/content dicts)\n"
        "  - Pre-formatted: text"
    )

#load and prepare dataset func
def load_and_prepare_dataset(config: Dict, tokenizer) -> Tuple[Dataset, Optional[Dataset]]:
    dataset_config = config['dataset']
    dataset_name = dataset_config['name']
    split = dataset_config.get('split', 'train')
    test_size = dataset_config.get('test_size', 0.1)
    seed = dataset_config.get('seed', 42)

    print(f"Loading dataset: {dataset_name}")
    
    dataset = load_dataset(dataset_name, split=split)
    print(f"Loaded {len(dataset)} examples")
    print(f"Dataset columns: {dataset.column_names}")
    
    dataset = detect_and_format_dataset(dataset, tokenizer, config)
    
    if test_size > 0:
        dataset_split = dataset.train_test_split(test_size=test_size, seed=seed)
        train_dataset = dataset_split['train']
        eval_dataset = dataset_split['test']
    else:
        train_dataset = dataset
        eval_dataset = None
    
    return train_dataset, eval_dataset

#setup chat template func
def setup_chat_template(tokenizer, config: Dict):
    chat_template_name = config.get('chat_template', {}).get('name', 'chatml')
    tokenizer = get_chat_template(tokenizer, chat_template=chat_template_name)
    return tokenizer

#main func
def main(config_path: str):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    print("=" * 60)
    print("Single-Turn Fine-Tuning with Unsloth")
    print("=" * 60)
    print(f"Loaded config from {config_path}")

    model_name = config['model']['name']
    print(f"Using model: {model_name}")

    # init wandb
    if config.get('training', {}).get('report_to') == 'wandb':
        if os.environ.get("WANDB_API_KEY"):
            wandb_config = config.get('wandb', {})
            wandb.init(
                entity=wandb_config.get('entity'),
                project=wandb_config.get('project'),
                name=config['training'].get('run_name', f"SFT-SingleTurn-{model_name}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"),
                tags=wandb_config.get('tags', []) + ['unsloth', 'single-turn'],
                notes=wandb_config.get('notes', 'Single-turn fine-tuning'),
                config=config
            )
            print("Initialized Weights & Biases logging.")
        else:
            print("WANDB_API_KEY not found. Skipping W&B logging.")
            config['training']['report_to'] = None

    #load model n tokenizer
    print("\nLoading model and tokenizer...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name,
        max_seq_length=config['dataset']['max_length'],
        dtype=None,
        load_in_4bit=config.get('quantization', {}).get('load_in_4bit', True),
        trust_remote_code=config.get('model', {}).get('trust_remote_code', False)
    )

    #lora conf
    lora_config = config.get('lora', {})
    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_config['r'],
        lora_alpha=lora_config['lora_alpha'],
        lora_dropout=lora_config['lora_dropout'],
        target_modules=lora_config['target_modules'],
        use_gradient_checkpointing="unsloth",
        use_rslora=False,
        loftq_config=None
    )
    
    print(f"Maximum sequence length: {model.max_seq_length}")
    print(f"LoRA: r={lora_config['r']}, alpha={lora_config['lora_alpha']}, dropout={lora_config['lora_dropout']}")
    print(f"Target modules: {lora_config['target_modules']}")

    #setup tokenizer padding
    padding_side = config.get('tokenizer', {}).get('padding_side', 'left')
    tokenizer.padding_side = padding_side
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print(f"Tokenizer: padding_side={padding_side}, pad_token={tokenizer.pad_token}")

    tokenizer = setup_chat_template(tokenizer, config)
    print("Chat template configured.")

    print("\nPreparing dataset for single-turn training...")
    train_dataset, eval_dataset = load_and_prepare_dataset(config, tokenizer)
    
    print(f"Training examples: {len(train_dataset)}")
    if eval_dataset:
        print(f"Evaluation examples: {len(eval_dataset)}")

    print("\n" + "-" * 40)
    print("Sample formatted text:")
    print("-" * 40)
    sample_text = train_dataset[0]['text']
    print(sample_text[:800] + "..." if len(sample_text) > 800 else sample_text)
    print("-" * 40)

    #sft conf
    train_config = config['training']
    output_dir = train_config['output_dir']
    
    sft_config = SFTConfig(
        output_dir=output_dir,
        run_name=train_config.get('run_name', f"SFT-SingleTurn-{datetime.now().strftime('%Y%m%d-%H%M%S')}"),
        per_device_train_batch_size=train_config['per_device_train_batch_size'],
        per_device_eval_batch_size=train_config['per_device_eval_batch_size'],
        gradient_accumulation_steps=train_config['gradient_accumulation_steps'],
        num_train_epochs=train_config['num_train_epochs'],
        learning_rate=train_config['learning_rate'],
        warmup_ratio=train_config['warmup_ratio'],
        optim=train_config.get('optim', 'adamw_8bit'),
        weight_decay=train_config['weight_decay'],
        lr_scheduler_type=train_config['lr_scheduler_type'],
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),        
        logging_steps=train_config['logging_steps'],
        eval_strategy=train_config.get('eval_strategy', 'steps'),
        eval_steps=train_config['eval_steps'],
        save_strategy=train_config['save_strategy'],
        save_steps=train_config['save_steps'],
        save_total_limit=train_config['save_total_limit'],        
        load_best_model_at_end=train_config['load_best_model_at_end'],
        metric_for_best_model=train_config.get('metric_for_best_model', 'eval_loss'),
        greater_is_better=train_config.get('greater_is_better', False),
        seed=config['dataset'].get('seed', 42),
        report_to=train_config.get('report_to', None),
    )

    #callbacks bcs training so takes time
    callbacks = []
    early_stopping_patience = train_config.get('early_stopping_patience')
    if early_stopping_patience:
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=early_stopping_patience))
        print(f"Early stopping enabled with patience: {early_stopping_patience}")

    print("\nInitializing SFT Trainer...")
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        args=sft_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        max_seq_length=model.max_seq_length,
        callbacks=callbacks if callbacks else None,
        dataset_num_proc=train_config.get('dataset_num_proc', 2),
    )

    #train on responses only for single-turn
    trainer = train_on_responses_only(
        trainer,
        instruction_part="<|im_start|>user\n",
        response_part="<|im_start|>assistant\n",
    )
    print("Applied train_on_responses_only (instruction masking)")
    print("\n" + "=" * 60)
    print("Training Configuration Summary")
    print("=" * 60)
    print(f"Model: {model_name}")
    print(f"Dataset: {config['dataset']['name']}")
    print(f"Training Type: Single-Turn")
    print(f"Max Sequence Length: {model.max_seq_length}")
    print(f"LoRA Rank (r): {lora_config['r']}")
    print(f"LoRA Alpha: {lora_config['lora_alpha']}")
    print(f"Batch Size: {train_config['per_device_train_batch_size']}")
    print(f"Gradient Accumulation: {train_config['gradient_accumulation_steps']}")
    print(f"Effective Batch Size: {train_config['per_device_train_batch_size'] * train_config['gradient_accumulation_steps']}")
    print(f"Epochs: {train_config['num_train_epochs']}")
    print(f"Learning Rate: {train_config['learning_rate']}")
    print(f"Output: {output_dir}")
    print("=" * 60)

    try:
        print("\nStarting training...")
        trainer_stats = trainer.train()
        print("\n[DONE] Training completed successfully!")
        print(f"Final training loss: {trainer_stats.training_loss:.4f}")
    except torch.cuda.OutOfMemoryError:
        print("\n[FAIL] ERROR: Out of Memory. Try reducing batch size or max_length.")
        sys.exit(1)
    except Exception as e:
        print(f"\n[FAIL] ERROR: {e}")
        sys.exit(1)

    print("\nSaving model and tokenizer...")
    final_model_dir = config['paths']['final_model_dir']
    model.save_pretrained(final_model_dir)
    tokenizer.save_pretrained(final_model_dir)
    print(f"[DONE] Model saved to {final_model_dir}")

    if trainer.state.best_metric is not None:
        print(f"Best {sft_config.metric_for_best_model}: {trainer.state.best_metric:.4f}")

    #up to wandb
    if wandb.run is not None:
        print("\nUploading model to W&B...")
        try:
            artifact_name = config.get('wandb', {}).get('artifact_name', f"model-{wandb.run.name}")
            artifact = wandb.Artifact(
                name=artifact_name,
                type="model",
                description=f"Single-turn fine-tuned model from {wandb.run.name}",
                metadata={
                    "model_name": model_name,
                    "dataset": config['dataset']['name'],
                    "training_type": "single-turn",
                    "lora_r": lora_config['r'],
                    "lora_alpha": lora_config['lora_alpha'],
                    "epochs": train_config['num_train_epochs'],
                    "learning_rate": train_config['learning_rate'],
                    "training_loss": trainer_stats.training_loss,
                    "best_eval_loss": trainer.state.best_metric,
                }
            )
            artifact.add_dir(final_model_dir)
            wandb.log_artifact(artifact)
            print(f"[DONE] Model uploaded as artifact: {artifact_name}")
        except Exception as e:
            print(f"[WARN] Warning: Failed to upload to W&B: {e}")
        
        wandb.finish()
        print("W&B run finished.")

    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Single-Turn Fine-Tuning with Unsloth",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Supported Dataset Formats:
  - Alpaca: instruction, input (optional), output
  - Prompt-Completion: prompt, completion  
  - Q&A: question, answer
  - Simple: input, output
  - Messages: messages (list of role/content dicts)
  - Pre-formatted: text

Example:
  python train_unsloth_single_turn.py --config configs/sft_single_turn.yaml
        """
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML configuration file"
    )
    args = parser.parse_args()

    if not os.path.isfile(args.config):
        print(f"ERROR: Config file {args.config} does not exist.")
        sys.exit(1)
    
    main(args.config)
