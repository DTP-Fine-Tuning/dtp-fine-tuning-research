"""
Gradio Interface for Fine-tuned Models
This script provides an interactive web interface for chatting with fine-tuned models.
Supports: Llama, Qwen, Mistral, Gemma, Phi, and other model families.
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


# ============================================================================
# Model Family Configuration Registry
# ============================================================================

MODEL_FAMILIES = {
    'llama': {
        'patterns': ['llama', 'llama-2', 'llama-3', 'codellama'],
        'default_system_message': 'You are a helpful, respectful and honest assistant.',
        'stop_strings': ['<|eot_id|>', '<|end_of_text|>', '<|start_header_id|>user<|end_header_id|>'],
    },
    'qwen': {
        'patterns': ['qwen', 'qwen2', 'qwen2.5', 'qwen3'],
        'default_system_message': 'You are a helpful assistant.',
        'stop_strings': ['<|im_end|>', '<|endoftext|>', '\n\nuser', '\n\nUser:'],
    },
    'mistral': {
        'patterns': ['mistral', 'mixtral'],
        'default_system_message': 'You are a helpful assistant.',
        'stop_strings': ['</s>', '[INST]', '\n\nuser', '\n\nUser:'],
    },
    'gemma': {
        'patterns': ['gemma'],
        'default_system_message': 'You are a helpful assistant.',
        'stop_strings': ['<end_of_turn>', '<eos>', '\n\nuser', '\n\nUser:'],
    },
    'phi': {
        'patterns': ['phi-2', 'phi-3'],
        'default_system_message': 'You are a helpful assistant.',
        'stop_strings': ['<|endoftext|>', '<|end|>', '\n\nuser', '\n\nUser:'],
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


class ChatInterface:
    """Main class for multi-model chat interface with Gradio"""

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
            model_path: Path to the fine-tuned model (LoRA adapter or merged model)
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
        self.model_family = None
        self.stop_strings = []
        self.load_model_config()

        # Initialize model and tokenizer
        self.model = None
        self.tokenizer = None
        self.streamer = None

        # Load model
        self.load_model(load_in_4bit)

        # Get stop strings after tokenizer is loaded
        self.setup_stop_tokens()

        logger.info(f"Model loaded successfully from {model_path}")
        logger.info(f"Detected model family: {self.model_family or 'unknown'}")
        logger.info(f"Stop strings: {self.stop_strings}")
    
    def load_model_config(self):
        """Load model configuration from training_info.json if available"""
        training_info_path = self.model_path / "training_info.json"

        if training_info_path.exists():
            with open(training_info_path, 'r') as f:
                training_info = json.load(f)

            if not self.base_model_name:
                self.base_model_name = training_info.get("model_name")

            self.model_family = training_info.get("model_family")
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

            self.chat_template_config = {}
            logger.warning("training_info.json not found, using default configuration")

        # Detect model family if not already set
        if not self.model_family and self.base_model_name:
            self.model_family = detect_model_family(self.base_model_name)

        # Set default system message based on model family
        if not self.chat_template_config.get("system_message"):
            if self.model_family and self.model_family in MODEL_FAMILIES:
                self.chat_template_config["system_message"] = MODEL_FAMILIES[self.model_family]['default_system_message']
            else:
                self.chat_template_config["system_message"] = "You are a helpful assistant."

        if "use_system_message" not in self.chat_template_config:
            self.chat_template_config["use_system_message"] = True
    
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

    def setup_stop_tokens(self):
        """Setup stop strings based on model family and tokenizer"""
        # Get stop strings from model family config
        if self.model_family and self.model_family in MODEL_FAMILIES:
            self.stop_strings = MODEL_FAMILIES[self.model_family]['stop_strings'].copy()
        else:
            # Default stop strings
            self.stop_strings = ['\n\nuser', '\n\nUser:', 'user\n', 'User\n']

        # Add tokenizer's EOS token if available
        if hasattr(self.tokenizer, 'eos_token') and self.tokenizer.eos_token:
            if self.tokenizer.eos_token not in self.stop_strings:
                self.stop_strings.append(self.tokenizer.eos_token)

        logger.info(f"Configured stop strings: {self.stop_strings}")

    def format_prompt(
        self,
        message: str,
        history: List[Tuple[str, str]],
        system_message: Optional[str] = None
    ) -> str:
        """
        Format the conversation history and current message into a prompt using tokenizer's chat template.

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

        # Build messages list
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

        # Use tokenizer's chat template if available
        if hasattr(self.tokenizer, 'apply_chat_template') and self.tokenizer.chat_template:
            try:
                prompt = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
                return prompt
            except Exception as e:
                logger.warning(f"Failed to use chat template: {e}, falling back to simple format")

        # Fallback: simple format
        prompt = ""
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            if role == "system":
                prompt += f"System: {content}\n\n"
            elif role == "user":
                prompt += f"User: {content}\n\n"
            elif role == "assistant":
                prompt += f"Assistant: {content}\n\n"

        prompt += "Assistant: "
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

        # Debug: Log prompt (first 200 chars and last 200 chars)
        logger.info(f"Prompt start: {prompt[:200]}")
        logger.info(f"Prompt end: {prompt[-200:]}")
        logger.info(f"Total prompt length: {len(prompt)} chars")

        # Tokenize input
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048
        )
        
        if self.device == "cuda":
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        # Generate response
        generation_kwargs = {
            "inputs": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "repetition_penalty": repetition_penalty,
            "do_sample": temperature > 0,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }
        
        if stream:
            generation_kwargs["streamer"] = self.streamer

            # Start generation in a separate thread
            thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
            thread.start()

            # Stream output with stop string detection
            generated_text = ""
            min_length = 10  # Minimum characters before checking stop strings

            for new_text in self.streamer:
                generated_text += new_text

                # Only check for stop strings after generating minimum length
                # and only check if stop string appears near the end
                should_stop = False
                if len(generated_text.strip()) >= min_length:
                    for stop_str in self.stop_strings:
                        # Check if stop string is at the end or near the end (last 50 chars)
                        check_region = generated_text[-50:] if len(generated_text) > 50 else generated_text
                        if stop_str in check_region:
                            # Find the position and truncate
                            idx = generated_text.rfind(stop_str)
                            if idx != -1:
                                generated_text = generated_text[:idx].strip()
                                should_stop = True
                                break

                # Only yield if we have content
                if generated_text.strip():
                    yield generated_text

                if should_stop:
                    break

            thread.join()

            # Final yield with cleaned text
            if generated_text.strip():
                yield generated_text
        else:
            with torch.no_grad():
                output = self.model.generate(**generation_kwargs)

            # Decode output
            generated_text = self.tokenizer.decode(
                output[0][len(inputs["input_ids"][0]):],
                skip_special_tokens=True
            )

            logger.info(f"Generated (before stop check): {generated_text[:100]}...")

            # Check for stop strings in non-streaming mode (only near the end)
            for stop_str in self.stop_strings:
                # Check if stop string is in the last 100 characters
                check_region = generated_text[-100:] if len(generated_text) > 100 else generated_text
                if stop_str in check_region:
                    idx = generated_text.rfind(stop_str)
                    if idx != -1:
                        generated_text = generated_text[:idx].strip()
                        logger.info(f"Stopped at: {stop_str}")
                        break

            logger.info(f"Final output length: {len(generated_text)}")
            yield generated_text
    
    def clear_conversation(self):
        """Clear the conversation history"""
        return None, []


def create_gradio_interface(model_path: str, **kwargs):
    """
    Create a Gradio interface for the chat model

    Args:
        model_path: Path to the fine-tuned model
        **kwargs: Additional arguments for ChatInterface

    Returns:
        Gradio interface
    """
    # Initialize chat interface
    chat_interface = ChatInterface(model_path, **kwargs)
    
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
    model_display_name = chat_interface.model_family.upper() if chat_interface.model_family else "LLM"
    with gr.Blocks(title=f"{model_display_name} Chat Interface", theme=gr.themes.Soft()) as interface:
        gr.Markdown(
            f"""
            # {model_display_name} Fine-tuned Model Chat Interface

            This interface allows you to interact with your fine-tuned model.
            **Model Family**: {chat_interface.model_family or 'Generic'}
            **Base Model**: {chat_interface.base_model_name}

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
                    maximum=2048,
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
                - **Model Family**: {chat_interface.model_family or 'Unknown'}
                - **Device**: {chat_interface.device}
                - **Quantization**: {'4-bit' if kwargs.get('load_in_4bit', True) else 'Full precision'}
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

    parser = argparse.ArgumentParser(
        description="Gradio interface for fine-tuned models (Llama, Qwen, Mistral, Gemma, etc.)"
    )
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
    
    args = parser.parse_args()
    
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
