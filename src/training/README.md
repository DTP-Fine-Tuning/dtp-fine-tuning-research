**Overview**
- **File**: `src/training/train_sft_generic_v2.py` : Generic supervised fine-tuning (SFT) training script that uses Hugging Face `Trainer` with a custom instruction-masking data collator and LoRA adapters. Supports multiple model families (Llama, Qwen, Mistral, Gemma, Phi, etc.) and 4-bit quantization via `bitsandbytes`.

**Purpose**
- Train language models for instruction-following (assistant) behavior while computing loss only on assistant responses (completion-only loss). The script: loads model/tokenizer, applies LoRA adapters, prepares datasets using a chat template, tokenizes inputs, uses a custom data collator to mask labels, and trains using `transformers.Trainer`.

**Prerequisites**
- Python 3.8+
- Install repository dependencies (from repo `requirements.txt`):

```
transformers
datasets
peft
accelerate
torch
trl
wandb
```

- GPU recommended (supports CUDA and mixed precision). For 4-bit quantized training, ensure `bitsandbytes` is available via your `transformers`/bnb setup.

**Quick Start / Usage**
- Run training with a YAML config file (example path `configs/sft_llama3_2_1B_1k.yaml`):

```bash
python src/training/train_sft_generic_v2.py --config configs/sft_llama3_2_1B_1k.yaml
```

- The repository also contains helper scripts in `scripts/` (e.g. `run_training.sh`) to wrap common invocations.

**High-level Flow**
- Load YAML config
- Detect model family (heuristic lookup)
- Configure W&B environment (if present)
- Create tokenizer and (if missing) apply a chat template
- Create BitsAndBytes quantization config (4-bit optional)
- Load base model and prepare for k-bit training
- Create LoRA config and wrap model with PEFT
- Load dataset via `datasets.load_dataset`, split train/eval
- Convert examples into chat format using the tokenizer's chat template
- Tokenize (no padding) and then use a custom data collator to pad and mask labels
- Build `TrainingArguments` and callbacks
- Initialize `transformers.Trainer` and call `trainer.train()`
- Save LoRA adapter and tokenizer, optionally merge and save a fully merged model
- Optionally upload artifacts to Weights & Biases

**Key Files & Symbols**
- `src/training/train_sft_generic_v2.py` : main script (this file)
- `DataCollatorForCompletionOnly` : custom collator that masks labels so loss is computed only on assistant responses
- `get_response_template` : infers or reads the assistant response delimiter/template used to locate assistant spans in tokenized sequences
- `format_dataset_with_chat_template` : converts a variety of dataset schemas into the tokenizer's chat-format text
- `create_tokenizer`, `create_model`, `create_quantization_config`, `create_lora_config` : helper constructors
- `create_training_arguments` : maps YAML training fields to `transformers.TrainingArguments`
- `create_callbacks` : builds Trainer callbacks (early stopping, memory monitor, custom logging, save-best)

**Configuration (YAML) — Important Keys**
- `model`:
  - `name`: HF model identifier (e.g. `meta-llama/Llama-2-7b-chat-hf`)
  - `trust_remote_code` (optional)
  - `use_cache` (optional)

- `tokenizer`:
  - `padding_side` (optional)
  - `trust_remote_code` (optional)

- `quantization`:
  - `load_in_4bit`: true/false
  - `bnb_4bit_quant_type`: e.g. `nf4`
  - `bnb_4bit_compute_dtype`: `float16`/`bfloat16`/`float32`
  - `bnb_4bit_use_double_quant`: true/false

- `lora`:
  - `r`, `lora_alpha`, `lora_dropout`, `target_modules` (list), `bias`, `task_type`

- `dataset`:
  - `name`: dataset id for `datasets.load_dataset`
  - `split` (optional), `test_size` (optional), `seed`, `max_length`
  - Example formats supported: fields `messages`, `conversation(s)`, `text`, `instruction`+`output`, `input`+`output`.

- `training`:
  - `output_dir`, `per_device_train_batch_size`, `per_device_eval_batch_size`, `gradient_accumulation_steps`
  - `num_train_epochs`, `learning_rate`, `warmup_ratio`, `optim`, `weight_decay`, `lr_scheduler_type`
  - `fp16` / `bf16`, `logging_steps`, `eval_strategy`, `eval_steps`, `save_strategy`, `save_steps`, `save_total_limit`
  - `report_to`: `wandb` or `none`

- `callbacks`:
  - `early_stopping`: `enabled` + `patience`
  - `custom_logging`, `memory_monitor`, `save_best_model` toggles and params

- `paths`:
  - `final_model_dir` : where the final LoRA adapter (and tokenizer) will be saved

- `save_merged_model`: boolean to optionally merge LoRA weights into a base model and save separately

**Data Collator Behavior**
- `DataCollatorForCompletionOnly` tokenizes the configured `response_template` and (optionally) `instruction_template` and locates occurrence(s) of the response template within tokenized sequences.
- For each sequence, it sets all label positions to `ignore_index` (-100) except the token span corresponding to the assistant response. This ensures loss is calculated only on assistant output tokens.
- If the response template is not found in a sequence, the collator warns and masks all tokens (safety).
- Padding is applied in the collator; tokenization is performed earlier with `padding=False`.

**Model Family Detection & Chat Templates**
- The script contains a small `MODEL_FAMILIES` registry and `detect_model_family()` to auto-infer a family from the model name. This influences default `padding_side` and `response_template` inference.
- If the tokenizer lacks a `chat_template`, `setup_chat_template()` attempts to apply a default based on family or a custom template supplied in the config under `chat_template.template_string`.

**Callbacks**
- `CustomLoggingCallback`: prints loss and eval_loss as logs arrive
- `MemoryMonitorCallback`: prints CUDA memory stats every N steps
- `SaveBestModelCallback`: tracks and announces best eval loss
- `EarlyStoppingCallback` from `transformers` is added if enabled

**W&B Integration**
- Controlled by `training.report_to` and `wandb` keys in config.
- If W&B is enabled and `WANDB_API_KEY` is set, the script will attempt to upload the final model directory as an artifact.

**Common Troubleshooting & Tips**
- OutOfMemory: see suggestions printed by the script. Typical mitigations:
  - Reduce `dataset.max_length`
  - Reduce per-device `per_device_train_batch_size`
  - Decrease LoRA `r`
  - Increase `gradient_accumulation_steps`
  - Use 4-bit quantization (`quantization.load_in_4bit = true`) if supported
- If the response template is not matching assistant output, the loss may be masked away — verify `response_template` detection and consider setting `training.response_template` in config.
- If tokenizer padding token is missing, the script sets `pad_token` to `eos_token` automatically.
- If tokenizer does not support `apply_chat_template` safely, ensure `trust_remote_code` is enabled or supply formatted `text` in dataset.

**Minimal Example Config Snippet**
```yaml
model:
  name: "meta-llama/Llama-2-7b-chat-hf"
  trust_remote_code: true

tokenizer:
  trust_remote_code: true

quantization:
  load_in_4bit: true
  bnb_4bit_quant_type: nf4
  bnb_4bit_compute_dtype: float16
  bnb_4bit_use_double_quant: true

lora:
  r: 8
  lora_alpha: 32
  lora_dropout: 0.05
  target_modules: ["q_proj", "v_proj"]
  bias: none
  task_type: CAUSAL_LM

dataset:
  name: "my_custom_sft_dataset"
  test_size: 0.02
  max_length: 1024

training:
  output_dir: "outputs/exp1"
  per_device_train_batch_size: 4
  per_device_eval_batch_size: 4
  gradient_accumulation_steps: 8
  num_train_epochs: 3
  learning_rate: 2e-4
  fp16: true
  report_to: wandb

paths:
  final_model_dir: "outputs/exp1/final"

save_merged_model: true
```

**Outputs**
- The script writes the LoRA-adapter-compatible model files and tokenizer to `paths.final_model_dir`.
- A `training_info.json` is saved to the model directory with metadata about the run (model name, lora config, tokenizer info, dataset info, timestamp).
- Optionally, a merged full model directory is created and uploaded as a separate artifact when `save_merged_model: true`.

**Extending / Reuse**
- `DataCollatorForCompletionOnly` can be reused with other training pipelines that require completion-only loss.
- The model-family registry can be extended by adding family names and patterns to `MODEL_FAMILIES`.
- To add custom logging metrics, extend `create_callbacks()` and/or pass a `compute_metrics` callable to `Trainer`.

**Next Steps / Suggestions**
- Add an example YAML config under `configs/` for a minimal run (small model or toy dataset) to help new contributors validate the pipeline quickly.
- Add unit tests for the data collator behavior (e.g., ensure label masking correctly preserves assistant spans).
- Provide a small example dataset or a converter helper to create proper chat-format examples for common dataset shapes.

**Contact / Notes**
- When opening issues or requesting help, include the YAML config, a small sample dataset example (1-2 examples), CUDA memory logs (if OOM), and `training_info.json` if available.

---

Generated by repository tooling: documents `src/training/train_sft_generic_v2.py` behavior and configuration.

**Other scripts in `src/training/`**

- `deepeval_evaluation_gemini.py`
  - Purpose: Evaluation helper that integrates the DeepEval framework with Google Gemini as an evaluation LLM. Provides `GeminiModel` (a DeepEval-compatible wrapper around Gemini), and `MultiTurnEvaluator` which loads a fine-tuned model (merged or LoRA adapter) and runs single-turn or multi-turn evaluations using a set of metrics (correctness, coherence, relevancy, context consistency, hallucination, etc.).
  - Usage patterns:
    - As a library: import `MultiTurnEvaluator` from the module and call its methods from a script or notebook. Example:

      ```python
      from src.training.deepeval_evaluation_gemini import MultiTurnEvaluator

      evaluator = MultiTurnEvaluator(model_path='outputs/exp1/final', base_model_name='meta-llama/Llama-2-7b-chat-hf')
      results = evaluator.evaluate_on_dataset(dataset_name='my_dataset', test_size=100)
      print(results)
      ```

    - Before using Gemini-based evaluation, ensure you set `GEMINI_API_KEY` or `GOOGLE_API_KEY` in your environment. The module provides `validate_gemini_api()` and `create_sample_test_data()` helpers.
  - Notes: DeepEval classes and metrics are used; this script expects `deepeval` and `google.generativeai` to be available. It can evaluate both merged models and LoRA adapters (it loads adapters using `PeftModel` when adapter metadata is present).

- `gradio_inference.py`
  - Purpose: Lightweight Gradio web UI for live chat with a fine-tuned model. Supports streaming output via `TextIteratorStreamer`, streaming stop-detection using family-specific stop tokens, and saving conversation JSON files.
  - CLI usage:

    ```bash
    python src/training/gradio_inference.py --model-path outputs/exp1/final --base-model meta-llama/Llama-2-7b-chat-hf --port 7860
    ```

    Flags:
    - `--share`: create a public Gradio link
    - `--no-4bit`: disable 4-bit quantization
    - `--max-new-tokens`: override generation length

  - Key features:
    - Detects model family (Llama/Qwen/Mistral/Gemma/Phi) to apply stop-strings & default system messages.
    - Supports both merged-model and LoRA-adapter loading (detects `adapter_config.json`).
    - Streams generation chunks into the Gradio chat UI and truncates on stop tokens.
  - Notes: useful for manual qualitative checks and demos. Running this requires `gradio`, `transformers` and optional `bitsandbytes` if using 4-bit.

- `memory_estimator.py`
  - Purpose: Quick GPU memory estimator for planning fine-tuning experiments. Provides `estimate_memory_usage()` and `check_gpu_availability()` helpers and prints human-readable reports with recommendations.
  - Usage:

    ```bash
    python src/training/memory_estimator.py
    ```

  - Notes: The script uses simple heuristics (rough estimates) to help decide batch size, sequence length, LoRA rank and quantization choices. It is advisory — measure actual GPU usage during test runs to validate.

- `training_script_llama32_simple.py`
  - Purpose: A simpler SFT training wrapper that uses `trl.SFTTrainer` tailored for Llama 3.2-style chat formatting. Mirrors many behaviors of `train_sft_generic_v2.py` but uses `SFTTrainer` and a Llama-3.2 formatter.
  - CLI usage:

    ```bash
    python src/training/training_script_llama32_simple.py --config configs/sft_llama3_2_1B_1k.yaml
    ```

  - Key features:
    - Applies LoRA via `peft.LoraConfig` and `get_peft_model`.
    - Uses `SFTConfig` / `SFTTrainer` from `trl` (different trainer API compared to `transformers.Trainer`).
    - Formats conversations into the Llama 3.2 chat template (`<|begin_of_text|>`, `<|start_header_id|>...`).

- `training_script_qwen3_improved.py`
  - Purpose: Qwen3-oriented SFT training script with improved multi-turn support and tokenizer-template fallback handling. Uses `trl.SFTTrainer` and contains helpers to format data for Qwen3 (`<|im_start|>...<|im_end|>` style).
  - CLI usage:

    ```bash
    python src/training/training_script_qwen3_improved.py --config configs/sft_qwen3_1_7B_1k.yaml
    ```

  - Key features:
    - Supports both tokenizer-provided `chat_template` and manual Qwen3 formatting.
    - Saves `training_info.json` with metadata for later inference/saving.
    - Optionally merges and saves a merged model when `save_merged_model: true`.

**General notes for all scripts**
- All training scripts expect a YAML config file with keys similar to those described in the "Configuration (YAML) — Important Keys" section above (model, tokenizer, quantization, lora, dataset, training, callbacks, paths).
- Scripts use `bitsandbytes` (via `transformers` BitsAndBytesConfig) for 4-bit quantization — ensure your environment supports `bitsandbytes` and compatible CUDA toolkit.
- LoRA adapter handling: scripts typically detect whether the saved model directory is a LoRA adapter (presence of `adapter_config.json`) or a merged model and load accordingly.
- Logging & experiment tracking: scripts integrate with Weights & Biases when `training.report_to` or `wandb` config is set; they expect `WANDB_API_KEY` in environment for artifact uploads.

If you'd like, I can:
- add example YAML files into `configs/` for each script (Llama, Qwen, and a tiny toy config for quick smoke tests),
- add a minimal sample dataset or test harness under `notebooks/` or `experiments/` for validation,
- or run a dry-run (no-network, tiny dataset) to validate the pipeline locally and capture runtime notes.
