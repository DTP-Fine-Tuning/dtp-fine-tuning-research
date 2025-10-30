#!/usr/bin/env python3
"""
GPU Memory Estimator for LLM Fine-tuning
Helps estimate if your configuration will fit in available GPU memory
"""

import torch
import numpy as np

def estimate_memory_usage(
    model_params_billion=4.0,
    batch_size=1,
    sequence_length=1024,
    lora_rank=16,
    quantization_bits=4,
    gradient_accumulation_steps=8,
    optimizer="adamw_8bit"
):
    """
    Estimate GPU memory usage for fine-tuning
    
    Args:
        model_params_billion: Number of parameters in billions
        batch_size: Training batch size per device
        sequence_length: Maximum sequence length
        lora_rank: LoRA rank (r parameter)
        quantization_bits: Quantization (4, 8, or 16 for fp16)
        gradient_accumulation_steps: Gradient accumulation steps
        optimizer: Optimizer type
    
    Returns:
        Dictionary with memory estimates
    """
    
    # Constants
    bytes_per_param = {
        4: 0.5,   # 4-bit quantization
        8: 1.0,   # 8-bit quantization
        16: 2.0,  # fp16
        32: 4.0   # fp32
    }
    
    # Base model memory
    model_memory_gb = model_params_billion * bytes_per_param[quantization_bits]
    
    # LoRA parameters (approximate)
    # Assuming LoRA is applied to attention layers (q, k, v, o projections)
    # and MLP layers (gate, up, down projections)
    hidden_size = int(np.sqrt(model_params_billion * 1e9 / 32))  # Rough estimate
    num_layers = 32  # Typical for 4B model
    
    lora_params = (
        7 * num_layers * hidden_size * lora_rank * 2  # 7 modules, in/out projections
    ) / 1e9  # Convert to billions
    
    lora_memory_gb = lora_params * bytes_per_param[16]  # LoRA params usually in fp16
    
    # Optimizer states
    optimizer_multiplier = {
        "sgd": 0,
        "adamw_8bit": 0.5,
        "paged_adamw_8bit": 0.5,
        "adamw": 2.0,
        "paged_adamw_32bit": 2.0,
    }
    
    optimizer_memory_gb = lora_memory_gb * optimizer_multiplier.get(optimizer, 2.0)
    
    # Gradients memory (for LoRA parameters only)
    gradients_memory_gb = lora_memory_gb
    
    # Activation memory (rough estimate)
    # Depends on batch size, sequence length, and hidden size
    activation_memory_gb = (
        batch_size * sequence_length * hidden_size * 4 * bytes_per_param[16]
    ) / 1e9
    
    # If using gradient checkpointing, reduce activation memory significantly
    if True:  # Assuming gradient checkpointing is enabled
        activation_memory_gb *= 0.1
    
    # Total memory estimate
    total_memory_gb = (
        model_memory_gb +
        lora_memory_gb +
        optimizer_memory_gb +
        gradients_memory_gb +
        activation_memory_gb +
        1.0  # Overhead (CUDA kernels, etc.)
    )
    
    return {
        "model_memory_gb": model_memory_gb,
        "lora_memory_gb": lora_memory_gb,
        "optimizer_memory_gb": optimizer_memory_gb,
        "gradients_memory_gb": gradients_memory_gb,
        "activation_memory_gb": activation_memory_gb,
        "overhead_gb": 1.0,
        "total_memory_gb": total_memory_gb,
    }


def check_gpu_availability():
    """Check available GPU memory"""
    if not torch.cuda.is_available():
        return None
    
    total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
    allocated = torch.cuda.memory_allocated() / 1024**3
    reserved = torch.cuda.memory_reserved() / 1024**3
    free = torch.cuda.mem_get_info()[0] / 1024**3
    
    return {
        "total_gb": total_memory,
        "allocated_gb": allocated,
        "reserved_gb": reserved,
        "free_gb": free,
    }


def print_memory_report(config_name, estimates, available_memory=None):
    """Pretty print memory estimates"""
    print(f"\n{'='*60}")
    print(f"Memory Estimate for: {config_name}")
    print('='*60)
    
    print("\nMemory Breakdown:")
    print(f"  Base Model:        {estimates['model_memory_gb']:.2f} GB")
    print(f"  LoRA Parameters:   {estimates['lora_memory_gb']:.2f} GB")
    print(f"  Optimizer States:  {estimates['optimizer_memory_gb']:.2f} GB")
    print(f"  Gradients:         {estimates['gradients_memory_gb']:.2f} GB")
    print(f"  Activations:       {estimates['activation_memory_gb']:.2f} GB")
    print(f"  Overhead:          {estimates['overhead_gb']:.2f} GB")
    print(f"  {'─'*30}")
    print(f"  Total Required:    {estimates['total_memory_gb']:.2f} GB")
    
    if available_memory:
        print(f"\nAvailable GPU Memory: {available_memory:.2f} GB")
        if estimates['total_memory_gb'] <= available_memory:
            margin = available_memory - estimates['total_memory_gb']
            print(f"✅ Configuration should fit! (Margin: {margin:.2f} GB)")
        else:
            deficit = estimates['total_memory_gb'] - available_memory
            print(f"❌ Configuration exceeds available memory by {deficit:.2f} GB!")
            print("\nSuggestions to reduce memory:")
            print("  • Reduce batch size")
            print("  • Reduce sequence length")
            print("  • Reduce LoRA rank")
            print("  • Use more aggressive quantization")
            print("  • Use a smaller model")


def main():
    """Run memory estimates for different configurations"""
    
    # Check GPU availability
    gpu_info = check_gpu_availability()
    if gpu_info:
        print("\n" + "="*60)
        print("GPU Information")
        print("="*60)
        print(f"Total Memory:     {gpu_info['total_gb']:.2f} GB")
        print(f"Currently Free:   {gpu_info['free_gb']:.2f} GB")
        print(f"Already Reserved: {gpu_info['reserved_gb']:.2f} GB")
        available = gpu_info['free_gb']
    else:
        print("No GPU detected! Using 9.0 GB as target.")
        available = 9.0
    
    # Configuration 1: Original Qwen3-4B with aggressive optimization
    config1 = estimate_memory_usage(
        model_params_billion=4.0,
        batch_size=1,
        sequence_length=1024,
        lora_rank=16,
        quantization_bits=4,
        gradient_accumulation_steps=8,
        optimizer="paged_adamw_8bit"
    )
    print_memory_report("Qwen3-4B (Optimized)", config1, available)
    
    # Configuration 2: Ultra-lite with even more aggressive settings
    config2 = estimate_memory_usage(
        model_params_billion=4.0,
        batch_size=1,
        sequence_length=512,
        lora_rank=8,
        quantization_bits=4,
        gradient_accumulation_steps=16,
        optimizer="paged_adamw_8bit"
    )
    print_memory_report("Qwen3-4B (Ultra-Lite)", config2, available)
    
    # Configuration 3: Smaller model (Qwen 1.5B)
    config3 = estimate_memory_usage(
        model_params_billion=1.5,
        batch_size=1,
        sequence_length=512,
        lora_rank=8,
        quantization_bits=4,
        gradient_accumulation_steps=16,
        optimizer="paged_adamw_8bit"
    )
    print_memory_report("Qwen2.5-1.5B (Recommended)", config3, available)
    
    # Configuration 4: Smaller model with comfort margin
    config4 = estimate_memory_usage(
        model_params_billion=1.5,
        batch_size=1,
        sequence_length=1024,
        lora_rank=16,
        quantization_bits=4,
        gradient_accumulation_steps=8,
        optimizer="paged_adamw_8bit"
    )
    print_memory_report("Qwen2.5-1.5B (Comfortable)", config4, available)
    
    print("\n" + "="*60)
    print("Recommendations for ~9GB Available Memory:")
    print("="*60)
    print("\n1. START with Qwen2.5-1.5B model - much more stable")
    print("2. Use 4-bit quantization (load_in_4bit=True)")
    print("3. Keep batch_size=1, use gradient accumulation")
    print("4. Start with LoRA r=8, increase if memory allows")
    print("5. Monitor memory during training with callbacks")
    print("6. Clear cache regularly (torch.cuda.empty_cache())")
    print("\nIf training fails, progressively reduce:")
    print("  • max_length (1024 → 512 → 256)")
    print("  • LoRA rank (16 → 8 → 4)")
    print("  • Consider CPU offloading for optimizer states")


if __name__ == "__main__":
    main()