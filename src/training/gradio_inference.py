"""
Gradio Interface for Qwen3 Fine-tuned Model
This script provides an interactive web interface for chatting with the fine-tuned model.
"""

import os
import sys
import torch
import gradio as gr
import json
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TextIteratorStreamer
)
from peft import PeftModel, PeftConfig
from threading import Thread
import logging
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class Qwen3ChatInterface:
    """Main class for Qwen3 chat interface with Gradio"""
    
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
        repetition_penalty: float = 1.1
    ):
        """
        Initialize the chat interface
        
        Args:
            model_path: Path to the fine-tuned model (LoRA adapter)
            base_model_name: Base model name (if not provided, will try to read from training_info.json)
            load_in_4bit: Whether to load model in 4-bit quantization
            device: Device to use (cuda/cpu)
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            top_k: Top-k sampling parameter
            repetition_penalty: Repetition penalty
        """
        self.model_path = Path(model_path)
        self.device = device if torch.cuda.is_available() else "cpu"
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.repetition_penalty = repetition_penalty
        
        # Load model configuration
        self.base_model_name = base_model_name
        self.load_model_config()
        
        # Initialize model and tokenizer
        self.model = None
        self.tokenizer = None
        self.streamer = None
        
        # Load model
        self.load_model(load_in_4bit)
        
        logger.info(f"Model loaded successfully from {model_path}")
    
    def load_model_config(self):
        """Load model configuration from training_info.json if available"""
        training_info_path = self.model_path / "training_info.json"
        
        if training_info_path.exists():
            with open(training_info_path, 'r') as f:
                training_info = json.load(f)
                
            if not self.base_model_name:
                self.base_model_name = training_info.get("model_name")
            
            self.chat_template_config = training_info.get("chat_template", {})
            logger.info(f"Loaded training configuration from {training_info_path}")
        else:
            # Try to load from adapter config
            adapter_config_path = self.model_path / "adapter_config.json"
            if adapter_config_path.exists():
                with open(adapter_config_path, 'r') as f:
                    adapter_config = json.load(f)
                    if not self.base_model_name:
                        self.base_model_name = adapter_config.get("base_model_name_or_path")
            
            self.chat_template_config = {
                "system_message": "You are a helpful assistant.",
                "use_system_message": True
            }
            logger.warning("training_info.json not found, using default configuration")
    
    def load_model(self, load_in_4bit: bool = True):
        """Load the fine-tuned model and tokenizer"""
        logger.info(f"Loading base model: {self.base_model_name}")
        
        # Create quantization config if needed
        bnb_config = None
        if load_in_4bit and self.device == "cuda":
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            padding_side="left"
        )
        
        # Ensure padding token is set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
        # Check if it's a merged model or LoRA adapter
        is_merged = not (self.model_path / "adapter_config.json").exists()
        
        if is_merged:
            # Load merged model directly
            logger.info("Loading merged model...")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                quantization_config=bnb_config,
                device_map="auto" if self.device == "cuda" else None,
                trust_remote_code=True,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            )
        else:
            # Load base model and LoRA adapter
            logger.info("Loading base model with LoRA adapter...")
            
            # Load base model
            base_model = AutoModelForCausalLM.from_pretrained(
                self.base_model_name,
                quantization_config=bnb_config,
                device_map="auto" if self.device == "cuda" else None,
                trust_remote_code=True,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            )
            
            # Load LoRA adapter
            self.model = PeftModel.from_pretrained(
                base_model,
                self.model_path,
                device_map="auto" if self.device == "cuda" else None,
            )
        
        # Set model to evaluation mode
        self.model.eval()
        
        # Initialize streamer for streaming output
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
        """
        Format the conversation history and current message into a prompt
        
        Args:
            message: Current user message
            history: List of (user, assistant) message tuples
            system_message: System message to use
            
        Returns:
            Formatted prompt string
        """
        # Use system message from config or provided one
        if system_message is None:
            system_message = self.chat_template_config.get(
                "system_message",
                "You are a helpful assistant."
            )
        
        # Build messages list for chat template
        messages = []
        
        # Add system message
        if self.chat_template_config.get("use_system_message", True) and system_message:
            messages.append({"role": "system", "content": system_message})
        
        # Add conversation history
        for user_msg, assistant_msg in history:
            if user_msg:
                messages.append({"role": "user", "content": user_msg})
            if assistant_msg:
                messages.append({"role": "assistant", "content": assistant_msg})
        
        # Add current message
        messages.append({"role": "user", "content": message})
        
        # Use tokenizer's apply_chat_template for proper formatting
        # This ensures the prompt matches the training format exactly
        try:
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False  # Disable thinking for direct responses
            )
        except TypeError:
            # Fallback if enable_thinking is not supported
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        
        # Debug: log the generated prompt (truncated for readability)
        logger.debug(f"Generated prompt (last 500 chars): ...{prompt[-500:]}")
        
        return prompt
    
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
    ) -> str:
        """
        Generate a response to the user message
        
        Args:
            message: User message
            history: Conversation history
            system_message: System message
            temperature: Sampling temperature (overrides default)
            top_p: Top-p parameter (overrides default)
            top_k: Top-k parameter (overrides default)
            max_new_tokens: Max new tokens (overrides default)
            repetition_penalty: Repetition penalty (overrides default)
            stream: Whether to stream the response
            
        Returns:
            Generated response
        """
        # Use provided parameters or defaults
        temperature = temperature if temperature is not None else self.temperature
        top_p = top_p if top_p is not None else self.top_p
        top_k = top_k if top_k is not None else self.top_k
        max_new_tokens = max_new_tokens if max_new_tokens is not None else self.max_new_tokens
        repetition_penalty = repetition_penalty if repetition_penalty is not None else self.repetition_penalty
        
        # Format prompt
        prompt = self.format_prompt(message, history, system_message)
        
        # Tokenize input
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=8192
        )
        
        if self.device == "cuda":
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        # Generate response
        # Get all possible stop token IDs for Qwen3/ChatML format
        stop_token_ids = [self.tokenizer.eos_token_id]
        
        # Add <|im_end|> token if it exists (important for Qwen3)
        im_end_token_id = self.tokenizer.convert_tokens_to_ids("<|im_end|>")
        if im_end_token_id is not None and im_end_token_id != self.tokenizer.unk_token_id:
            stop_token_ids.append(im_end_token_id)
        
        # Add <|endoftext|> token if it exists
        endoftext_token_id = self.tokenizer.convert_tokens_to_ids("<|endoftext|>")
        if endoftext_token_id is not None and endoftext_token_id != self.tokenizer.unk_token_id:
            stop_token_ids.append(endoftext_token_id)
        
        # Remove duplicates and None values
        stop_token_ids = list(set([t for t in stop_token_ids if t is not None]))
        
        generation_kwargs = {
            "inputs": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
            "max_new_tokens": max_new_tokens,
            "temperature": temperature if temperature > 0 else 1.0,
            "top_p": top_p,
            "top_k": top_k,
            "repetition_penalty": repetition_penalty,
            "do_sample": temperature > 0,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": stop_token_ids,
        }
        
        if stream:
            generation_kwargs["streamer"] = self.streamer
            
            # Start generation in a separate thread
            thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
            thread.start()
            
            # Stream output
            generated_text = ""
            for new_text in self.streamer:
                generated_text += new_text
                # Clean up any remaining special tokens from streaming
                cleaned_text = self._clean_response(generated_text)
                yield cleaned_text
            
            thread.join()
        else:
            with torch.no_grad():
                output = self.model.generate(**generation_kwargs)
            
            # Decode output
            generated_text = self.tokenizer.decode(
                output[0][len(inputs["input_ids"][0]):],
                skip_special_tokens=True
            )
            
            # Clean up any remaining special tokens
            generated_text = self._clean_response(generated_text)
            
            yield generated_text
    
    def _clean_response(self, text: str) -> str:
        """Clean up response text by removing special tokens and artifacts"""
        import re
        
        # Remove any remaining ChatML-style special tokens
        text = re.sub(r'<\|im_start\|>.*?\n', '', text)
        text = re.sub(r'<\|im_end\|>', '', text)
        text = re.sub(r'<\|endoftext\|>', '', text)
        
        # Remove any thinking tags if they appear in output
        text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
        
        # Clean up extra whitespace
        text = text.strip()
        
        return text
    
    def clear_conversation(self):
        """Clear the conversation history"""
        return None, []


def create_gradio_interface(model_path: str, **kwargs):
    """
    Create a Gradio interface for the chat model
    
    Args:
        model_path: Path to the fine-tuned model
        **kwargs: Additional arguments for Qwen3ChatInterface
        
    Returns:
        Gradio interface
    """
    # Initialize chat interface
    chat_interface = Qwen3ChatInterface(model_path, **kwargs)
    
    # Define the chat function for Gradio
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
        """Chat function for Gradio interface"""
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
    
    # Create Gradio interface
    with gr.Blocks(title="Qwen3 Chat Interface", theme=gr.themes.Soft()) as interface:
        gr.Markdown(
            """
            # Qwen3 Fine-tuned Model Chat Interface
            
            This interface allows you to interact with your fine-tuned Qwen3 model.
            Adjust the parameters on the right to control the generation behavior.
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
                    minimum=0.0,
                    maximum=2.0,
                    value=0.7,
                    step=0.1,
                    info="Controls randomness in generation"
                )
                
                top_p = gr.Slider(
                    label="Top-p",
                    minimum=0.0,
                    maximum=1.0,
                    value=0.95,
                    step=0.05,
                    info="Nucleus sampling parameter"
                )
                
                top_k = gr.Slider(
                    label="Top-k",
                    minimum=1,
                    maximum=100,
                    value=50,
                    step=1,
                    info="Top-k sampling parameter"
                )
                
                max_new_tokens = gr.Slider(
                    label="Max New Tokens",
                    minimum=1,
                    maximum=8192,
                    value=512,
                    step=1,
                    info="Maximum number of tokens to generate"
                )
                
                repetition_penalty = gr.Slider(
                    label="Repetition Penalty",
                    minimum=1.0,
                    maximum=2.0,
                    value=1.1,
                    step=0.05,
                    info="Penalty for repeating tokens"
                )
                
                gr.Markdown("### Model Information")
                gr.Markdown(f"""
                - **Model Path**: {model_path}
                - **Base Model**: {chat_interface.base_model_name}
                - **Device**: {chat_interface.device}
                """)
        
        # Event handlers
        def user_submit(message, history):
            return "", history + [[message, None]]
        
        def bot_respond(history, system_msg, temp, top_p_val, top_k_val, max_tokens, rep_penalty):
            if not history or history[-1][1] is not None:
                return history
            
            message = history[-1][0]
            history_context = history[:-1]
            
            response = ""
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
                response = chunk
                history[-1][1] = response
                yield history
        
        def save_conversation(history):
            """Save conversation to a file"""
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"conversation_{timestamp}.json"
            
            conversation_data = {
                "timestamp": timestamp,
                "model_path": str(model_path),
                "conversation": history
            }
            
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(conversation_data, f, ensure_ascii=False, indent=2)
            
            return f"Conversation saved to {filename}"
        
        # Connect events
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
        
        clear.click(lambda: (None, []), None, [chatbot, chatbot], queue=False)
        
        save.click(
            save_conversation,
            [chatbot],
            None
        ).then(
            lambda x: gr.Info(x),
            None,
            None
        )
    
    return interface


def main():
    """Main function to run the Gradio interface"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Gradio interface for Qwen3 fine-tuned model")
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to the fine-tuned model directory"
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default=None,
        help="Base model name (if not provided, will read from training_info.json)"
    )
    parser.add_argument(
        "--share",
        action="store_true",
        help="Create a public Gradio link"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=7860,
        help="Port to run the Gradio interface on"
    )
    parser.add_argument(
        "--no-4bit",
        action="store_true",
        help="Disable 4-bit quantization"
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=512,
        help="Maximum number of tokens to generate"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging to see generated prompts"
    )
    
    args = parser.parse_args()
    
    # Set debug logging if requested
    if args.debug:
        logging.getLogger(__name__).setLevel(logging.DEBUG)
        logger.debug("Debug mode enabled - prompts will be logged")
    
    # Check if model path exists
    if not Path(args.model_path).exists():
        print(f"Error: Model path {args.model_path} does not exist")
        sys.exit(1)
    
    # Create and launch interface
    interface = create_gradio_interface(
        model_path=args.model_path,
        base_model_name=args.base_model,
        load_in_4bit=not args.no_4bit,
        max_new_tokens=args.max_new_tokens
    )
    
    print(f"\n{'='*50}")
    print(f"Launching Gradio interface...")
    print(f"Model: {args.model_path}")
    print(f"Port: {args.port}")
    print(f"Public link: {'Yes' if args.share else 'No'}")
    print(f"{'='*50}\n")
    
    interface.launch(
        share=args.share,
        server_port=args.port,
        server_name="0.0.0.0"
    )


if __name__ == "__main__":
    main()
