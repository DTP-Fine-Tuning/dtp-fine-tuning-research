"""
DeepEval Evaluation Script for Qwen3 Fine-tuned Model with Google Gemini Integration
This script evaluates the model using various metrics for multi-turn conversations.
Uses Google Gemini API instead of OpenAI for evaluation metrics.
"""

import os
import sys
import json
import torch
import yaml
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import pandas as pd
import logging
import google.generativeai as genai
from tenacity import retry, stop_after_attempt, wait_exponential

# DeepEval imports
from deepeval import evaluate, assert_test
from deepeval.metrics import (
    GEval,
    AnswerRelevancyMetric,
    FaithfulnessMetric,
    ContextualRelevancyMetric,
    HallucinationMetric,
    BiasMetric,
    ToxicityMetric,
    ConversationCompletenessMetric,
    ConversationRelevancyMetric,
    KnowledgeRetentionMetric
)
from deepeval.test_case import LLMTestCase, ConversationalTestCase, LLMTestCaseParams
from deepeval.dataset import EvaluationDataset, Golden

# Custom LLM model for DeepEval using Gemini
from deepeval.models import DeepEvalBaseLLM

# Model loading imports
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig
)
from peft import PeftModel
from datasets import load_dataset

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class GeminiModel(DeepEvalBaseLLM):
    """Custom Gemini model wrapper for DeepEval integration"""
    
    def __init__(
        self, 
        model_name: str = "gemini-1.5-flash",
        api_key: Optional[str] = None,
        temperature: float = 0.7,
        max_output_tokens: int = 2048
    ):
        """
        Initialize Gemini model for DeepEval
        
        Args:
            model_name: Gemini model to use (gemini-1.5-pro, gemini-1.5-flash, gemini-1.0-pro)
            api_key: Google API key for Gemini
            temperature: Temperature for generation
            max_output_tokens: Maximum output tokens
        """
        self.model_name = model_name
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        
        # Configure Gemini API
        if api_key:
            genai.configure(api_key=api_key)
        elif "GEMINI_API_KEY" in os.environ:
            genai.configure(api_key=os.environ["GEMINI_API_KEY"])
        elif "GOOGLE_API_KEY" in os.environ:
            genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
        else:
            raise ValueError(
                "Google Gemini API key not found. Please set GEMINI_API_KEY or GOOGLE_API_KEY "
                "environment variable or pass api_key parameter."
            )
        
        # Initialize the Gemini model
        self.model = genai.GenerativeModel(model_name)
        self.generation_config = genai.GenerationConfig(
            temperature=temperature,
            max_output_tokens=max_output_tokens,
        )
        
        logger.info(f"Initialized Gemini model: {model_name}")
    
    def load_model(self):
        """Load model - already done in __init__"""
        return self.model
    
    def get_model_name(self) -> str:
        """Return the model name"""
        return self.model_name
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def generate(self, prompt: str) -> str:
        """
        Generate response using Gemini API with retry logic
        
        Args:
            prompt: Input prompt for generation
            
        Returns:
            Generated text response
        """
        try:
            response = self.model.generate_content(
                prompt,
                generation_config=self.generation_config
            )
            
            # Handle safety filters
            if response.prompt_feedback and hasattr(response.prompt_feedback, 'block_reason'):
                logger.warning(f"Prompt blocked: {response.prompt_feedback.block_reason}")
                return "Response blocked due to safety filters."
            
            if response.text:
                return response.text
            else:
                logger.warning("Empty response from Gemini")
                return "Empty response generated."
                
        except Exception as e:
            logger.error(f"Error generating with Gemini: {str(e)}")
            raise
    
    async def a_generate(self, prompt: str) -> str:
        """
        Async generation for compatibility (delegates to sync version)
        
        Args:
            prompt: Input prompt for generation
            
        Returns:
            Generated text response
        """
        # For now, use sync version. Can be updated to use async client if needed
        return self.generate(prompt)


class MultiTurnEvaluator:
    """Evaluator for multi-turn conversation models using DeepEval with Gemini"""
    
    def __init__(
        self,
        model_path: str,
        base_model_name: Optional[str] = None,
        load_in_4bit: bool = True,
        device: str = "cuda",
        gemini_api_key: Optional[str] = None,
        gemini_model_name: str = "gemini-1.5-flash"
    ):
        """
        Initialize the evaluator with Gemini integration
        
        Args:
            model_path: Path to the fine-tuned model
            base_model_name: Base model name
            load_in_4bit: Whether to load in 4-bit quantization
            device: Device to use
            gemini_api_key: Google API key for Gemini evaluation metrics
            gemini_model_name: Gemini model to use for evaluation
        """
        self.model_path = Path(model_path)
        self.device = device if torch.cuda.is_available() else "cpu"
        
        # Set Gemini API key if provided
        if gemini_api_key:
            os.environ["GEMINI_API_KEY"] = gemini_api_key
        
        # Initialize Gemini model for evaluation
        self.gemini_model = GeminiModel(
            model_name=gemini_model_name,
            api_key=gemini_api_key
        )
        
        # Load model configuration
        self.base_model_name = base_model_name
        self.load_model_config()
        
        # Initialize model and tokenizer
        self.model = None
        self.tokenizer = None
        self.load_model(load_in_4bit)
        
        # Initialize metrics with Gemini
        self.metrics = self.initialize_metrics()
        
        logger.info(f"Evaluator initialized with model from {model_path}")
        logger.info(f"Using Gemini model: {gemini_model_name}")
    
    def load_model_config(self):
        """Load model configuration from training_info.json"""
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
    
    def load_model(self, load_in_4bit: bool = True):
        """Load the fine-tuned model and tokenizer"""
        logger.info(f"Loading model from {self.model_path}")
        
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
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
        # Check if it's a merged model or LoRA adapter
        is_merged = not (self.model_path / "adapter_config.json").exists()
        
        if is_merged:
            # Load merged model
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                quantization_config=bnb_config,
                device_map="auto" if self.device == "cuda" else None,
                trust_remote_code=True,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            )
        else:
            # Load base model with LoRA adapter
            base_model = AutoModelForCausalLM.from_pretrained(
                self.base_model_name,
                quantization_config=bnb_config,
                device_map="auto" if self.device == "cuda" else None,
                trust_remote_code=True,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            )
            
            self.model = PeftModel.from_pretrained(
                base_model,
                self.model_path,
                device_map="auto" if self.device == "cuda" else None,
            )
        
        self.model.eval()
        logger.info("Model loaded successfully!")
    
    def initialize_metrics(self) -> Dict[str, Any]:
        """Initialize evaluation metrics with Gemini model"""
        metrics = {}
        
        # General quality metrics using Gemini
        metrics['correctness'] = GEval(
            name="Correctness",
            criteria="Determine if the actual output correctly answers the input question.",
            evaluation_params=[
                LLMTestCaseParams.ACTUAL_OUTPUT,
                LLMTestCaseParams.EXPECTED_OUTPUT,
                LLMTestCaseParams.INPUT
            ],
            threshold=0.7,
            model=self.gemini_model,  # Use Gemini model
            strict_mode=False
        )
        
        metrics['coherence'] = GEval(
            name="Coherence",
            criteria="Evaluate if the response is coherent, well-structured, and flows naturally.",
            evaluation_params=[
                LLMTestCaseParams.ACTUAL_OUTPUT,
                LLMTestCaseParams.INPUT
            ],
            threshold=0.7,
            model=self.gemini_model,
            strict_mode=False
        )
        
        metrics['relevancy'] = GEval(
            name="Relevancy",
            criteria="Determine if the response is relevant to the user's question and stays on topic.",
            evaluation_params=[
                LLMTestCaseParams.ACTUAL_OUTPUT,
                LLMTestCaseParams.INPUT
            ],
            threshold=0.7,
            model=self.gemini_model,
            strict_mode=False
        )
        
        # Multi-turn specific metrics
        metrics['context_consistency'] = GEval(
            name="Context Consistency",
            criteria="Evaluate if the response is consistent with the conversation history and maintains context.",
            evaluation_params=[
                LLMTestCaseParams.ACTUAL_OUTPUT,
                LLMTestCaseParams.INPUT,
                LLMTestCaseParams.CONTEXT
            ],
            threshold=0.7,
            model=self.gemini_model,
            strict_mode=False
        )
        
        # Hallucination metric with Gemini
        try:
            metrics['hallucination'] = HallucinationMetric(
                threshold=0.3,  # Lower is better for hallucination
                model=self.gemini_model
            )
        except Exception as e:
            logger.warning(f"Could not initialize hallucination metric: {e}")
            # Fallback to custom GEval for hallucination
            metrics['hallucination'] = GEval(
                name="Hallucination Check",
                criteria="Check if the response contains made-up information not grounded in the context or facts.",
                evaluation_params=[
                    LLMTestCaseParams.ACTUAL_OUTPUT,
                    LLMTestCaseParams.CONTEXT,
                    LLMTestCaseParams.INPUT
                ],
                threshold=0.3,  # Lower threshold for hallucination
                model=self.gemini_model,
                strict_mode=False
            )
        
        logger.info(f"Initialized {len(metrics)} evaluation metrics with Gemini")
        return metrics
    
    def generate_response(self, prompt: str, max_new_tokens: int = 512) -> str:
        """Generate response from the fine-tuned model"""
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
        
        if self.device == "cuda":
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.7,
                do_sample=True,
                top_p=0.95,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Remove the input prompt from the response
        if prompt in response:
            response = response.replace(prompt, "").strip()
        
        return response
    
    def evaluate_single_turn(
        self, 
        test_data: List[Dict], 
        metrics_to_use: Optional[List[str]] = None
    ) -> Dict:
        """
        Evaluate single-turn conversations
        
        Args:
            test_data: List of test cases with 'input', 'expected_output', and 'context'
            metrics_to_use: Specific metrics to use
            
        Returns:
            Dictionary with evaluation results
        """
        results = {}
        test_cases = []
        
        # Select metrics
        if metrics_to_use:
            selected_metrics = [self.metrics[m] for m in metrics_to_use if m in self.metrics]
        else:
            selected_metrics = [
                self.metrics.get('correctness'),
                self.metrics.get('coherence'),
                self.metrics.get('relevancy')
            ]
            selected_metrics = [m for m in selected_metrics if m is not None]
        
        # Generate responses and create test cases
        for item in test_data:
            actual_output = self.generate_response(item['input'])
            
            test_case = LLMTestCase(
                input=item['input'],
                actual_output=actual_output,
                expected_output=item.get('expected_output', ''),
                context=item.get('context', [])
            )
            test_cases.append(test_case)
            
            logger.info(f"Generated response for: {item['input'][:50]}...")
        
        # Evaluate with selected metrics
        for metric in selected_metrics:
            try:
                logger.info(f"Evaluating with metric: {metric.name if hasattr(metric, 'name') else metric.__class__.__name__}")
                
                scores = []
                for test_case in test_cases:
                    metric.measure(test_case)
                    scores.append(metric.score)
                
                avg_score = sum(scores) / len(scores) if scores else 0
                metric_name = metric.name if hasattr(metric, 'name') else metric.__class__.__name__
                results[metric_name] = avg_score
                
                logger.info(f"{metric_name}: {avg_score:.3f}")
                
            except Exception as e:
                logger.error(f"Error evaluating with metric {metric}: {e}")
        
        return results
    
    def evaluate_multi_turn(
        self, 
        conversations: List[List[Dict]], 
        metrics_to_use: Optional[List[str]] = None
    ) -> Dict:
        """
        Evaluate multi-turn conversations
        
        Args:
            conversations: List of conversations, each being a list of turns
            metrics_to_use: Specific metrics to use
            
        Returns:
            Dictionary with evaluation results
        """
        results = {}
        
        # Select metrics for multi-turn evaluation
        if metrics_to_use:
            # Filter for multi-turn relevant metrics
            multi_turn_metrics = ['context_consistency', 'hallucination']
            selected_metrics = [
                self.metrics[m] for m in metrics_to_use 
                if m in self.metrics and m in multi_turn_metrics
            ]
        else:
            selected_metrics = [
                self.metrics.get('context_consistency'),
                self.metrics.get('hallucination')
            ]
            selected_metrics = [m for m in selected_metrics if m is not None]
        
        all_scores = {metric.name if hasattr(metric, 'name') else metric.__class__.__name__: [] 
                     for metric in selected_metrics}
        
        for conversation in conversations:
            context = []
            
            for turn in conversation:
                # Build context from previous turns
                if 'user' in turn:
                    user_input = turn['user']
                    
                    # Generate model response
                    if context:
                        # Format with context
                        prompt = self.format_conversation_prompt(context, user_input)
                    else:
                        prompt = user_input
                    
                    actual_output = self.generate_response(prompt)
                    
                    # Create test case for this turn
                    test_case = LLMTestCase(
                        input=user_input,
                        actual_output=actual_output,
                        expected_output=turn.get('assistant', ''),
                        context=context.copy()
                    )
                    
                    # Evaluate this turn
                    for metric in selected_metrics:
                        try:
                            metric.measure(test_case)
                            metric_name = metric.name if hasattr(metric, 'name') else metric.__class__.__name__
                            all_scores[metric_name].append(metric.score)
                        except Exception as e:
                            logger.error(f"Error evaluating turn with {metric}: {e}")
                    
                    # Update context for next turn
                    context.append(f"User: {user_input}")
                    context.append(f"Assistant: {actual_output}")
        
        # Calculate average scores
        for metric_name, scores in all_scores.items():
            if scores:
                results[metric_name] = sum(scores) / len(scores)
                logger.info(f"{metric_name}: {results[metric_name]:.3f}")
        
        return results
    
    def format_conversation_prompt(self, context: List[str], current_input: str) -> str:
        """Format conversation history and current input into a prompt"""
        prompt_parts = []
        
        # Add system message if configured
        if self.chat_template_config.get('use_system_message'):
            system_msg = self.chat_template_config.get(
                'system_message',
                'You are a helpful assistant.'
            )
            prompt_parts.append(f"System: {system_msg}")
        
        # Add conversation history
        prompt_parts.extend(context)
        
        # Add current input
        prompt_parts.append(f"User: {current_input}")
        prompt_parts.append("Assistant:")
        
        return "\n".join(prompt_parts)
    
    def evaluate_on_dataset(
        self,
        dataset_name: Optional[str] = None,
        dataset_path: Optional[str] = None,
        test_size: int = 100,
        save_results: bool = True,
        output_dir: str = "./evaluation_results"
    ) -> Dict:
        """
        Evaluate on a dataset
        
        Args:
            dataset_name: HuggingFace dataset name
            dataset_path: Local dataset path
            test_size: Number of samples to evaluate
            save_results: Whether to save results
            output_dir: Directory to save results
            
        Returns:
            Dictionary with evaluation results
        """
        # Load dataset
        if dataset_name:
            dataset = load_dataset(dataset_name, split=f"train[:{test_size}]")
        elif dataset_path:
            # Load from local file
            if dataset_path.endswith('.json'):
                with open(dataset_path, 'r') as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        dataset = Dataset.from_list(data[:test_size])
                    else:
                        dataset = Dataset.from_dict(data)
        else:
            raise ValueError("Either dataset_name or dataset_path must be provided")
        
        # Prepare test data
        test_data = []
        for item in dataset:
            test_case = {
                'input': item.get('input', item.get('question', item.get('text', ''))),
                'expected_output': item.get('output', item.get('answer', item.get('response', ''))),
                'context': item.get('context', [])
            }
            test_data.append(test_case)
        
        # Run evaluation
        results = self.evaluate_single_turn(test_data)
        
        # Save results if requested
        if save_results:
            self.save_results({'single_turn': results}, output_dir)
        
        return results
    
    def save_results(self, results: Dict, output_dir: str):
        """Save evaluation results to files"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save as JSON
        json_path = output_path / f"evaluation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results saved to {json_path}")
        
        # Save summary
        summary = self.create_summary(results)
        summary_path = output_path / "evaluation_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Save markdown report
        self.create_markdown_report(results, output_path / "evaluation_report.md")
    
    def create_summary(self, results: Dict) -> Dict:
        """Create a summary of evaluation results"""
        summary = {}
        
        for eval_type, metrics in results.items():
            if isinstance(metrics, dict):
                summary[eval_type] = {}
                for metric_name, value in metrics.items():
                    if isinstance(value, (int, float)):
                        summary[eval_type][metric_name] = round(value, 3)
        
        return summary
    
    def create_markdown_report(self, results: Dict, output_path: Path):
        """Create a markdown report of evaluation results"""
        report = []
        report.append(f"# Evaluation Report\n")
        report.append(f"**Generated on**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        report.append(f"**Model**: {self.model_path}\n")
        report.append(f"**Evaluation Model**: {self.gemini_model.get_model_name()}\n\n")
        
        # Single-turn results
        if 'single_turn' in results:
            report.append("## Single-Turn Evaluation\n")
            single_results = results['single_turn']
            
            if isinstance(single_results, dict):
                for metric_name, metric_results in single_results.items():
                    report.append(f"### {metric_name}\n")
                    if isinstance(metric_results, dict):
                        for key, value in metric_results.items():
                            report.append(f"- **{key}**: {value}\n")
                    else:
                        report.append(f"- **Score**: {metric_results}\n")
                    report.append("\n")
        
        # Multi-turn results
        if 'multi_turn' in results:
            report.append("## Multi-Turn Evaluation\n")
            multi_results = results['multi_turn']
            
            if isinstance(multi_results, dict):
                for metric_name, metric_results in multi_results.items():
                    report.append(f"### {metric_name}\n")
                    if isinstance(metric_results, dict):
                        for key, value in metric_results.items():
                            report.append(f"- **{key}**: {value}\n")
                    else:
                        report.append(f"- **Score**: {metric_results}\n")
                    report.append("\n")
        
        # Write report
        with open(output_path, 'w') as f:
            f.write(''.join(report))


def create_sample_test_data():
    """Create sample test data for evaluation"""
    single_turn_data = [
        {
            "input": "Apa ibukota Indonesia?",
            "expected_output": "Ibukota Indonesia adalah Jakarta.",
            "context": ["Indonesia adalah negara di Asia Tenggara."]
        },
        {
            "input": "Jelaskan fotosintesis dengan bahasa sederhana.",
            "expected_output": "Fotosintesis adalah proses di mana tumbuhan menggunakan sinar matahari, air, dan karbon dioksida untuk membuat makanan berupa gula dan menghasilkan oksigen.",
            "context": []
        },
        {
            "input": "Apa manfaat olahraga teratur?",
            "expected_output": "Olahraga teratur meningkatkan kesehatan jantung, memperkuat otot dan tulang, membantu menjaga berat badan sehat, mengurangi stres, memperbaiki mood, dan meningkatkan kualitas hidup secara keseluruhan.",
            "context": []
        },
        {
            "input": "Bagaimana cara membuat kopi yang enak?",
            "expected_output": "Untuk membuat kopi yang enak, gunakan biji kopi berkualitas yang baru digiling, perbandingan air dan kopi yang tepat (sekitar 1:15), air dengan suhu 90-96°C, dan waktu seduh yang sesuai dengan metode yang digunakan.",
            "context": []
        },
        {
            "input": "Apa itu kecerdasan buatan?",
            "expected_output": "Kecerdasan buatan (AI) adalah teknologi yang memungkinkan mesin atau komputer untuk meniru kemampuan kecerdasan manusia seperti pembelajaran, penalaran, dan pemecahan masalah.",
            "context": []
        }
    ]
    
    multi_turn_data = [
        [
            {
                "user": "Halo! Saya sedang merencanakan perjalanan ke Bali.",
                "assistant": "Wah, menarik sekali! Bali adalah destinasi wisata yang indah. Kapan Anda berencana untuk berkunjung?"
            },
            {
                "user": "Saya berencana pergi bulan Juli. Bagaimana cuacanya saat itu?",
                "assistant": "Juli adalah musim kering di Bali yang berlangsung dari April hingga Oktober. Anda bisa mengharapkan hari-hari cerah dengan suhu sekitar 27°C, kelembaban rendah, dan curah hujan minimal. Ini adalah salah satu waktu terbaik untuk berkunjung!"
            },
            {
                "user": "Bagus! Apa saja tempat yang wajib dikunjungi?",
                "assistant": "Beberapa tempat yang wajib dikunjungi di Bali termasuk Pura Tanah Lot untuk melihat matahari terbenam, Ubud untuk budaya dan sawah terasering, Pura Uluwatu dengan pemandangan tebing, Tegallalang Rice Terraces, dan pantai-pantai indah di Seminyak atau Nusa Dua."
            }
        ],
        [
            {
                "user": "Bisakah kamu membantu saya memahami machine learning?",
                "assistant": "Tentu! Machine learning adalah cabang dari kecerdasan buatan di mana komputer belajar dari data untuk membuat prediksi atau keputusan tanpa diprogram secara eksplisit untuk setiap tugas spesifik."
            },
            {
                "user": "Bagaimana cara kerjanya?",
                "assistant": "Algoritma machine learning bekerja dengan menemukan pola dalam data training. Mereka menyesuaikan parameter internal berdasarkan contoh-contoh, secara bertahap meningkatkan kemampuan mereka untuk membuat prediksi akurat pada data baru yang belum pernah dilihat sebelumnya."
            },
            {
                "user": "Apa saja aplikasinya di dunia nyata?",
                "assistant": "Machine learning digunakan dalam banyak aplikasi: sistem rekomendasi (Netflix, Amazon), filter spam email, asisten suara (Siri, Alexa), deteksi penipuan di perbankan, diagnosis medis, kendaraan otonom, dan penerjemahan bahasa."
            }
        ],
        [
            {
                "user": "Saya ingin belajar memasak. Dari mana sebaiknya memulai?",
                "assistant": "Bagus sekali! Untuk pemula, mulailah dengan mempelajari teknik dasar seperti memotong, menumis, dan merebus. Pelajari resep-resep sederhana seperti nasi goreng, telur dadar, atau sup sayuran yang tidak memerlukan banyak bahan atau teknik rumit."
            },
            {
                "user": "Peralatan apa saja yang penting untuk dimiliki?",
                "assistant": "Peralatan dasar yang penting: pisau chef yang tajam, talenan, wajan atau penggorengan, panci dengan berbagai ukuran, spatula, sendok kayu, mangkuk pencampur, dan alat pengukur. Dengan peralatan dasar ini, Anda sudah bisa membuat berbagai hidangan."
            },
            {
                "user": "Ada tips untuk membuat masakan lebih enak?",
                "assistant": "Beberapa tips penting: selalu gunakan bahan segar berkualitas, jangan lupa bumbui dengan garam secukupnya, cicipi makanan saat memasak untuk menyesuaikan rasa, panaskan wajan terlebih dahulu sebelum memasak, dan jangan terburu-buru - beri waktu untuk setiap tahap memasak."
            }
        ]
    ]
    
    return single_turn_data, multi_turn_data


def validate_gemini_api():
    """Validate Gemini API configuration"""
    if "GEMINI_API_KEY" not in os.environ and "GOOGLE_API_KEY" not in os.environ:
        logger.warning(
            "Gemini API key not found in environment variables.\n"
            "Please set either GEMINI_API_KEY or GOOGLE_API_KEY environment variable.\n"
            "You can get an API key from: https://makersuite.google.com/app/apikey"
        )
        return False
    
    try:
        # Try to initialize Gemini
        api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        genai.configure(api_key=api_key)
        
        # Test with a simple generation
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content("Hello, this is a test.")
        
        if response.text:
            logger.info("Gemini API validated successfully!")
            return True
        else:
            logger.error("Gemini API test failed: No response generated")
            return False
            
    except Exception as e:
        logger.error(f"Gemini API validation failed: {str(e)}")
        return False


def main():
    """Main evaluation function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate Qwen3 fine-tuned model using DeepEval with Gemini")
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to the fine-tuned model"
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default=None,
        help="Base model name"
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default=None,
        help="HuggingFace dataset name for evaluation"
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        default=None,
        help="Local dataset path for evaluation"
    )
    parser.add_argument(
        "--use-sample-data",
        action="store_true",
        help="Use sample test data instead of a dataset"
    )
    parser.add_argument(
        "--test-size",
        type=int,
        default=100,
        help="Number of samples to evaluate"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./evaluation_results",
        help="Directory to save evaluation results"
    )
    parser.add_argument(
        "--no-4bit",
        action="store_true",
        help="Disable 4-bit quantization"
    )
    parser.add_argument(
        "--metrics",
        type=str,
        nargs="+",
        default=None,
        help="Specific metrics to use for evaluation"
    )
    parser.add_argument(
        "--gemini-model",
        type=str,
        default="gemini-1.5-flash",
        choices=["gemini-1.5-pro", "gemini-1.5-flash", "gemini-1.0-pro"],
        help="Gemini model to use for evaluation (default: gemini-1.5-flash)"
    )
    parser.add_argument(
        "--gemini-api-key",
        type=str,
        default=None,
        help="Google API key for Gemini (can also use GEMINI_API_KEY or GOOGLE_API_KEY env var)"
    )
    
    args = parser.parse_args()
    
    # Check if model path exists
    if not Path(args.model_path).exists():
        print(f"Error: Model path {args.model_path} does not exist")
        sys.exit(1)
    
    # Validate Gemini API
    print(f"\n{'='*50}")
    print(f"Validating Gemini API configuration...")
    print(f"{'='*50}\n")
    
    if args.gemini_api_key:
        os.environ["GEMINI_API_KEY"] = args.gemini_api_key
    
    if not validate_gemini_api():
        print("\nError: Gemini API validation failed.")
        print("Please ensure you have set the GEMINI_API_KEY or GOOGLE_API_KEY environment variable.")
        print("Get your API key from: https://makersuite.google.com/app/apikey")
        sys.exit(1)
    
    # Initialize evaluator
    print(f"\n{'='*50}")
    print(f"Initializing evaluator...")
    print(f"Model: {args.model_path}")
    print(f"Gemini Model: {args.gemini_model}")
    print(f"{'='*50}\n")
    
    evaluator = MultiTurnEvaluator(
        model_path=args.model_path,
        base_model_name=args.base_model,
        load_in_4bit=not args.no_4bit,
        gemini_api_key=args.gemini_api_key,
        gemini_model_name=args.gemini_model
    )
    
    # Run evaluation
    if args.use_sample_data:
        print("Using sample test data for evaluation...")
        single_turn_data, multi_turn_data = create_sample_test_data()
        
        results = {}
        
        # Evaluate single-turn
        print("\nEvaluating single-turn conversations...")
        single_results = evaluator.evaluate_single_turn(
            single_turn_data,
            metrics_to_use=args.metrics
        )
        results['single_turn'] = single_results
        
        # Evaluate multi-turn
        print("\nEvaluating multi-turn conversations...")
        multi_results = evaluator.evaluate_multi_turn(
            multi_turn_data,
            metrics_to_use=args.metrics
        )
        results['multi_turn'] = multi_results
        
        # Save results
        evaluator.save_results(results, args.output_dir)
        
    else:
        if not args.dataset_name and not args.dataset_path:
            print("Error: Either --dataset-name, --dataset-path, or --use-sample-data must be provided")
            sys.exit(1)
        
        print(f"Evaluating on dataset...")
        results = evaluator.evaluate_on_dataset(
            dataset_name=args.dataset_name,
            dataset_path=args.dataset_path,
            test_size=args.test_size,
            save_results=True,
            output_dir=args.output_dir
        )
    
    # Print summary
    print(f"\n{'='*50}")
    print(f"Evaluation completed!")
    print(f"Results saved to: {args.output_dir}")
    print(f"Using Gemini Model: {args.gemini_model}")
    print(f"{'='*50}\n")
    
    # Print basic metrics summary
    if 'single_turn' in results:
        print("Single-Turn Results Summary:")
        for metric, value in results['single_turn'].items():
            if isinstance(value, (int, float)):
                print(f"  - {metric}: {value:.3f}")
    
    if 'multi_turn' in results:
        print("\nMulti-Turn Results Summary:")
        for metric, value in results['multi_turn'].items():
            if isinstance(value, (int, float)):
                print(f"  - {metric}: {value:.3f}")


if __name__ == "__main__":
    main()
