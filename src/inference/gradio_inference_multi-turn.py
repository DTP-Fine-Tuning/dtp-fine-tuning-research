#gradio interface script for diploy
#Author: Tim 2 - DTP Fine Tuning

import os
import sys
import torch
import gradio as gr
import json
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Generator
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TextIteratorStreamer
)
from peft import PeftModel
from threading import Thread
import logging
from datetime import datetime
import re
import traceback

#logging conf
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

#main class
class DiployChatInterface:
    
    def __init__(
        self,
        model_path: str,
        base_model_name: Optional[str] = None,
        load_in_4bit: bool = True,
        device: str = "cuda",
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.95,
        top_k: int = 50,
        repetition_penalty: float = 1.1,
        use_auth_token: Optional[str] = None
    ):
        #init
        self.model_path_str = model_path
        self.is_hub_model = "/" in model_path and not os.path.exists(model_path)
        self.use_auth_token = use_auth_token
        
        if not self.is_hub_model:
            self.model_path = Path(model_path)
            if not self.model_path.exists():
                raise ValueError(f"Local model path {model_path} does not exist")
        else:
            self.model_path = model_path
            logger.info(f"Detected HuggingFace Hub model: {model_path}")
        
        if torch.cuda.is_available() and device.startswith("cuda"):
            self.device = device
        else:
            self.device = "cpu"
            logger.warning("CUDA not available, using CPU")
        
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.repetition_penalty = repetition_penalty
        
        #load model config
        self.base_model_name = base_model_name
        self.load_model_config()
        
        #init model and tokenizer
        self.model = None
        self.tokenizer = None
        self.streamer = None
        
        #load model
        self.load_model(load_in_4bit)
        
        logger.info(f"Model loaded successfully from {model_path}")
    
    #func for check training info
    def load_model_config(self):
        if self.is_hub_model:
            from huggingface_hub import hf_hub_download
            try:
                training_info_path = hf_hub_download(
                    repo_id=self.model_path,
                    filename="training_info.json",
                    token=self.use_auth_token
                )
                with open(training_info_path, 'r', encoding='utf-8') as f:
                    training_info = json.load(f)
                    
                if not self.base_model_name:
                    self.base_model_name = training_info.get("model_name")
                
                self.chat_template_config = training_info.get("chat_template", {})
                logger.info(f"Loaded training configuration from HuggingFace Hub")
            except Exception as e:
                logger.warning(f"Could not load training_info.json from Hub: {e}")
                try:
                    adapter_config_path = hf_hub_download(
                        repo_id=self.model_path,
                        filename="adapter_config.json",
                        token=self.use_auth_token
                    )
                    with open(adapter_config_path, 'r', encoding='utf-8') as f:
                        adapter_config = json.load(f)
                        if not self.base_model_name:
                            self.base_model_name = adapter_config.get("base_model_name_or_path")
                except Exception as e2:
                    logger.warning(f"Could not load adapter_config.json from Hub: {e2}")
                
                self.chat_template_config = {
                    "system_message": "You are a helpful assistant.",
                    "use_system_message": True
                }
        else:
            training_info_path = self.model_path / "training_info.json"
            
            if training_info_path.exists():
                with open(training_info_path, 'r', encoding='utf-8') as f:
                    training_info = json.load(f)
                    
                if not self.base_model_name:
                    self.base_model_name = training_info.get("model_name")
                
                self.chat_template_config = training_info.get("chat_template", {})
                logger.info(f"Loaded training configuration from {training_info_path}")
            else:
                adapter_config_path = self.model_path / "adapter_config.json"
                if adapter_config_path.exists():
                    with open(adapter_config_path, 'r', encoding='utf-8') as f:
                        adapter_config = json.load(f)
                        if not self.base_model_name:
                            self.base_model_name = adapter_config.get("base_model_name_or_path")
                
                self.chat_template_config = {
                    "system_message": "You are a helpful assistant.",
                    "use_system_message": True
                }
                logger.warning("training_info.json not found, using default configuration")
    
    #func for load model
    def load_model(self, load_in_4bit: bool = True):
        logger.info(f"Loading model from: {self.model_path_str}")
        
        #quant first
        bnb_config = None
        if load_in_4bit and self.device.startswith("cuda"):
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,  
                bnb_4bit_use_double_quant=True,
            )
        
        #load tokenizer
        logger.info("Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path if self.is_hub_model else str(self.model_path),
            trust_remote_code=True,
            padding_side="left",
            token=self.use_auth_token
        )
        
        #ensure pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
        #for checking merged or not
        if self.is_hub_model:
            from huggingface_hub import list_repo_files
            try:
                repo_files = list_repo_files(self.model_path, token=self.use_auth_token)
                is_merged = "adapter_config.json" not in repo_files
            except Exception as e:
                logger.warning(f"Could not list repo files, assuming merged model: {e}")
                is_merged = True
        else:
            is_merged = not (self.model_path / "adapter_config.json").exists()
        
        device_map_config = "auto" if self.device.startswith("cuda") else None
        torch_dtype = torch.bfloat16 if self.device.startswith("cuda") else torch.float16
        
        if is_merged:
            logger.info("Loading merged model...")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path if self.is_hub_model else str(self.model_path),
                quantization_config=bnb_config,
                device_map=device_map_config,
                trust_remote_code=True,
                torch_dtype=torch_dtype,
                token=self.use_auth_token
            )
        else:
            logger.info("Loading base model with LoRA adapter...")
            
            if not self.base_model_name:
                raise ValueError("base_model_name is required for LoRA adapter models")
            
            base_model = AutoModelForCausalLM.from_pretrained(
                self.base_model_name,
                quantization_config=bnb_config,
                device_map=device_map_config,
                trust_remote_code=True,
                torch_dtype=torch_dtype,
                token=self.use_auth_token
            )
            
            self.model = PeftModel.from_pretrained(
                base_model,
                self.model_path if self.is_hub_model else str(self.model_path),
                device_map=device_map_config,
                token=self.use_auth_token
            )
        
        self.model.eval()
        
        #init straemer
        self.streamer = TextIteratorStreamer(
            self.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True
        )
        
        logger.info("Model loaded successfully!")
    
    def format_prompt(
        self,
        message: str,
        history: List[Tuple[str, str]],
        system_message: Optional[str] = None
    ) -> str:
        #format conversation into prompt
        if system_message is None:
            system_message = self.chat_template_config.get(
                "system_message",
                "You are a helpful assistant."
            )
        
        messages = []
        
        if self.chat_template_config.get("use_system_message", True) and system_message:
            messages.append({"role": "system", "content": system_message})
        
        for user_msg, assistant_msg in history:
            if user_msg:
                messages.append({"role": "user", "content": user_msg})
            if assistant_msg:
                messages.append({"role": "assistant", "content": assistant_msg})
        
        messages.append({"role": "user", "content": message})
        
        try:
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )

        #catch error using manual formatting
        except Exception as e:
            logger.error(f"Error applying chat template: {e}")
            prompt = self._manual_format_prompt(messages)
        
        logger.debug(f"Generated prompt (last 500 chars): ...{prompt[-500:]}")
        
        return prompt
    
    #func for manual formatting
    def _manual_format_prompt(self, messages: List[Dict[str, str]]) -> str:
        prompt = ""
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            if role == "system":
                prompt += f"<|im_start|>system\n{content}<|im_end|>\n"
            elif role == "user":
                prompt += f"<|im_start|>user\n{content}<|im_end|>\n"
            elif role == "assistant":
                prompt += f"<|im_start|>assistant\n{content}<|im_end|>\n"
        
        prompt += "<|im_start|>assistant\n"
        return prompt
    
    #func for generating response
    def generate_response(
        self,
        message: str,
        history: List[Tuple[str, str]],
        system_message: Optional[str] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        max_new_tokens: Optional[int] = None,
        repetition_penalty: Optional[float] = None,
        stream: bool = True
    ) -> Generator[str, None, None]:
        
        #parameters
        temperature = temperature if temperature is not None else self.temperature
        top_p = top_p if top_p is not None else self.top_p
        top_k = top_k if top_k is not None else self.top_k
        max_new_tokens = max_new_tokens if max_new_tokens is not None else self.max_new_tokens
        repetition_penalty = repetition_penalty if repetition_penalty is not None else self.repetition_penalty
        
        try:
            prompt = self.format_prompt(message, history, system_message)
            
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=8192
            )
            
            if self.device.startswith("cuda"):
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            stop_token_ids = [self.tokenizer.eos_token_id]
            
            im_end_token_id = self.tokenizer.convert_tokens_to_ids("<|im_end|>")
            if im_end_token_id is not None and im_end_token_id != self.tokenizer.unk_token_id:
                stop_token_ids.append(im_end_token_id)
            
            endoftext_token_id = self.tokenizer.convert_tokens_to_ids("<|endoftext|>")
            if endoftext_token_id is not None and endoftext_token_id != self.tokenizer.unk_token_id:
                stop_token_ids.append(endoftext_token_id)
            
            stop_token_ids = list(set([t for t in stop_token_ids if t is not None]))
            
            generation_kwargs = {
                "inputs": inputs["input_ids"],
                "attention_mask": inputs["attention_mask"],
                "max_new_tokens": max_new_tokens,
                "temperature": max(temperature, 0.01),  
                "top_p": top_p,
                "top_k": top_k,
                "repetition_penalty": repetition_penalty,
                "do_sample": temperature > 0,
                "pad_token_id": self.tokenizer.pad_token_id,
                "eos_token_id": stop_token_ids,
            }
            
            if stream:
                generation_kwargs["streamer"] = self.streamer
                
                #error handling in thread
                generation_error = [None]  
                
                def generate_with_error_handling():
                    try:
                        self.model.generate(**generation_kwargs)
                    except Exception as e:
                        generation_error[0] = e
                        logger.error(f"Generation error: {e}")
                
                thread = Thread(target=generate_with_error_handling)
                thread.start()
                
                generated_text = ""
                try:
                    for new_text in self.streamer:
                        if generation_error[0]:
                            raise generation_error[0]
                        
                        generated_text += new_text
                        cleaned_text = self._clean_response(generated_text)
                        yield cleaned_text
                except Exception as e:
                    logger.error(f"Streaming error: {e}")
                    yield f"Error during generation: {str(e)}"
                finally:
                    thread.join(timeout=5)
                    
            else:
                with torch.no_grad():
                    output = self.model.generate(**generation_kwargs)
                
                generated_text = self.tokenizer.decode(
                    output[0][len(inputs["input_ids"][0]):],
                    skip_special_tokens=True
                )
                
                generated_text = self._clean_response(generated_text)
                yield generated_text
                
        except Exception as e:
            logger.error(f"Error in generate_response: {e}\n{traceback.format_exc()}")
            yield f"An error occurred: {str(e)}"
    
    #func for cleaning response
    def _clean_response(self, text: str) -> str:
        text = re.sub(r'^<\|im_start\|>\w+\n', '', text)
        text = re.sub(r'<\|im_end\|>$', '', text)
        text = re.sub(r'<\|endoftext\|>$', '', text)
        text = re.sub(r'<think>.*?</think>\s*', '', text, flags=re.DOTALL)
        text = text.strip()
        
        return text

#func for creating diploy interface
def create_gradio_interface(model_path: str, **kwargs):
    try:
        chat_interface = DiployChatInterface(model_path, **kwargs)
    except Exception as e:
        logger.error(f"Failed to initialize chat interface: {e}\n{traceback.format_exc()}")
        raise
    
    def chat_fn(
        message: str,
        history: List[Tuple[str, str]],
        system_message: str,
        temperature: float,
        top_p: float,
        top_k: int,
        max_new_tokens: int,
        repetition_penalty: float
    ):
        #chat func
        if not message or not message.strip():
            return ""
        
        response = ""
        for chunk in chat_interface.generate_response(
            message=message,
            history=history,
            system_message=system_message,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_new_tokens=max_new_tokens,
            repetition_penalty=repetition_penalty,
            stream=True
        ):
            response = chunk
            yield response
    
    #create diploy interface
    with gr.Blocks(title="Diploy Chat Interface", theme=gr.themes.Soft()) as interface:
        gr.Markdown(
            """
            # 🤖 Diploy Fine-tuned Model Chat Interface
            
            Interact with your fine-tuned Diploy model. Adjust parameters to control generation.
            """
        )
        
        with gr.Row():
            with gr.Column(scale=7):
                chatbot = gr.Chatbot(
                    height=500,
                    show_label=False,
                    elem_id="chatbot",
                    bubble_full_width=False
                )
                
                with gr.Row():
                    msg = gr.Textbox(
                        label="Message",
                        placeholder="Type your message here...",
                        lines=2,
                        scale=9
                    )
                    submit = gr.Button("Send", variant="primary", scale=1)
                
                with gr.Row():
                    clear = gr.Button("Clear", variant="secondary")
                    save = gr.Button("Save Chat", variant="secondary")
            
            with gr.Column(scale=3):
                gr.Markdown("### Generation Settings")
                
                system_message = gr.Textbox(
                    label="System Message",
                    value=chat_interface.chat_template_config.get(
                        "system_message",
                        "You are a helpful assistant."
                    ),
                    lines=3
                )
                
                temperature = gr.Slider(
                    label="Temperature",
                    minimum=0.01,  
                    maximum=2.0,
                    value=0.3,
                    step=0.05,
                    info="Controls randomness (0.01 = focused, 2.0 = creative)"
                )
                
                top_p = gr.Slider(
                    label="Top-p (Nucleus Sampling)",
                    minimum=0.1,
                    maximum=1.0,
                    value=0.9,
                    step=0.05,
                    info="Probability mass for nucleus sampling"
                )
                
                top_k = gr.Slider(
                    label="Top-k",
                    minimum=1,
                    maximum=100,
                    value=30,
                    step=1,
                    info="Number of top tokens to consider"
                )
                
                max_new_tokens = gr.Slider(
                    label="Max New Tokens",
                    minimum=64,
                    maximum=8192,
                    value=4096,
                    step=64,
                    info="Maximum response length"
                )
                
                repetition_penalty = gr.Slider(
                    label="Repetition Penalty",
                    minimum=1.0,
                    maximum=2.0,
                    value=1.1,
                    step=0.05,
                    info="Penalty for repeating tokens (1.0 = no penalty)"
                )
                
                gr.Markdown("###  Model Information")
                model_source = " HuggingFace Hub" if chat_interface.is_hub_model else " Local"
                gr.Markdown(f"""
                - **Source**: {model_source}
                - **Path**: `{model_path}`
                - **Base**: {chat_interface.base_model_name or "N/A (Merged)"}
                - **Device**: {chat_interface.device}
                - **Quantization**: {"4-bit" if kwargs.get('load_in_4bit', True) else "Full Precision"}
                """)
        
        #event handlers
        def user_submit(message, history):
            if not message or not message.strip():
                return message, history
            return "", history + [[message, None]]
        
        def bot_respond(history, system_msg, temp, top_p_val, top_k_val, max_tokens, rep_penalty):
            if not history or history[-1][1] is not None:
                return history
            
            message = history[-1][0]
            history_context = history[:-1]
            
            for chunk in chat_interface.generate_response(
                message=message,
                history=history_context,
                system_message=system_msg,
                temperature=temp,
                top_p=top_p_val,
                top_k=top_k_val,
                max_new_tokens=max_tokens,
                repetition_penalty=rep_penalty,
                stream=True
            ):
                history[-1][1] = chunk
                yield history
        
        #func for saving conversation
        def save_conversation(history):
            if not history:
                return "No conversation to save!"
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"conversation_{timestamp}.json"
            
            conversation_data = {
                "timestamp": timestamp,
                "model_path": str(model_path),
                "conversation": history
            }
            
            try:
                with open(filename, 'w', encoding='utf-8') as f:
                    json.dump(conversation_data, f, ensure_ascii=False, indent=2)
                
                return f"Conversation saved to {filename}"
            except Exception as e:
                return f"Error saving: {str(e)}"
        
        #connect events
        msg.submit(
            user_submit,
            [msg, chatbot],
            [msg, chatbot],
            queue=False
        ).then(
            bot_respond,
            [chatbot, system_message, temperature, top_p, top_k, max_new_tokens, repetition_penalty],
            chatbot
        )
        
        submit.click(
            user_submit,
            [msg, chatbot],
            [msg, chatbot],
            queue=False
        ).then(
            bot_respond,
            [chatbot, system_message, temperature, top_p, top_k, max_new_tokens, repetition_penalty],
            chatbot
        )
        
        clear.click(lambda: None, None, chatbot, queue=False)
        
        save_output = gr.Textbox(visible=False)
        save.click(
            save_conversation,
            [chatbot],
            save_output
        ).then(
            lambda x: gr.Info(x) if x else None,
            save_output,
            None
        )
    
    return interface

#main func
def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Gradio interface for Diploy fine-tuned model")
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to model (local or HuggingFace Hub ID like 'username/model-name')"
    )
    parser.add_argument("--base-model", type=str, default=None, help="Base model name")
    parser.add_argument("--share", action="store_true", help="Create public Gradio link")
    parser.add_argument("--port", type=int, default=7860, help="Port number")
    parser.add_argument("--no-4bit", action="store_true", help="Disable 4-bit quantization")
    parser.add_argument("--max-new-tokens", type=int, default=512, help="Max tokens to generate")
    parser.add_argument("--auth-token", type=str, default=None, help="HuggingFace token")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    
    args = parser.parse_args()
    
    if args.debug:
        logging.getLogger(__name__).setLevel(logging.DEBUG)
    
    auth_token = args.auth_token or os.environ.get("HF_TOKEN")
    
    is_hub_model = "/" in args.model_path and not os.path.exists(args.model_path)
    if not is_hub_model and not Path(args.model_path).exists():
        logger.error(f"Local model path {args.model_path} does not exist")
        sys.exit(1)
    
    try:
        interface = create_gradio_interface(
            model_path=args.model_path,
            base_model_name=args.base_model,
            load_in_4bit=not args.no_4bit,
            max_new_tokens=args.max_new_tokens,
            use_auth_token=auth_token
        )
        
        print(f"\n{'='*60}")
        print(f"Launching Gradio Interface")
        print(f"{'='*60}")
        print(f"Model: {args.model_path}")
        print(f"Port: {args.port}")
        print(f"Public: {'Yes' if args.share else 'No'}")
        print(f"{'='*60}\n")
        
        interface.launch(
            share=args.share,
            server_port=args.port,
            server_name="0.0.0.0"
        )
    except Exception as e:
        logger.error(f"Failed to launch interface: {e}\n{traceback.format_exc()}")
        sys.exit(1)


if __name__ == "__main__":
    main()