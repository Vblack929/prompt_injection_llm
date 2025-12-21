#!/usr/bin/env python3
"""
DPO Training for Qwen3 0.6B Model
Simple implementation with automatic wandb logging
"""

import os
import logging
import argparse
import torch
import matplotlib.pyplot as plt
from datetime import datetime
from typing import Dict, List, Any

from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import LoraConfig, get_peft_model, TaskType

from loss import CustomDPOTrainer
from custom_dataset import DPODataset
from setup import setup
from generate_dpo_data import extract_model_name, generate_dpo_data

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Environment setup
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def dpo_collate_fn(features: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Simple collate function for DPO training"""
    batch = {}
    
    # Stack tensor fields
    tensor_keys = [
        "prompt_input_ids", "prompt_attention_mask",
        "chosen_input_ids", "chosen_attention_mask", "chosen_labels",
        "rejected_input_ids", "rejected_attention_mask", "rejected_labels"
    ]
    
    for key in tensor_keys:
        if key in features[0]:
            batch[key] = torch.stack([f[key] for f in features])
    
    # Keep string fields as lists
    # string_keys = ["question", "prompt", "chosen", "rejected"]
    # for key in string_keys:
    #     if key in features[0]:
    #         batch[key] = [f[key] for f in features]
    
    return batch


def create_ref_model(model_path: str):
    """Create reference model for DPO training"""
    ref_model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype='auto',
        device_map='auto',
        trust_remote_code=True
    )
    ref_model.eval()
    return ref_model


def setup_model_and_tokenizer(
    model_path: str = "Qwen/Qwen3-0.6B",
    use_lora: bool = True, 
    lora_r: int = 16, 
    lora_alpha: int = 32, 
    lora_dropout: float = 0.1,
    lora_target_modules: List[str] = None,
    lora_bias: str = "none"
):
    """
    Setup model and tokenizer with optional LoRA
    
    Args:
        model_path: HuggingFace model name or path
        use_lora: Whether to use LoRA for efficient fine-tuning
        lora_r: LoRA rank (lower = fewer parameters)
        lora_alpha: LoRA alpha parameter (scaling factor)
        lora_dropout: LoRA dropout rate
        lora_target_modules: List of module names to apply LoRA to (default: ["q_proj", "v_proj"])
        lora_bias: Bias handling ("none", "all", "lora_only")
    
    Returns:
        tuple: (model, tokenizer)
    """
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype='auto',
        device_map='auto',
        trust_remote_code=True
    )
    
    if use_lora:
        # Default target modules for Qwen/Llama models
        if lora_target_modules is None:
            lora_target_modules = ["q_proj", "v_proj"]
        
        # Configure LoRA
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=lora_target_modules,
            bias=lora_bias,
        )
        model = get_peft_model(model, lora_config)
        
        # Print trainable parameters
        logger.info("LoRA configuration:")
        logger.info(f"  Rank (r): {lora_r}")
        logger.info(f"  Alpha: {lora_alpha}")
        logger.info(f"  Dropout: {lora_dropout}")
        logger.info(f"  Target modules: {lora_target_modules}")
        logger.info(f"  Bias: {lora_bias}")
        model.print_trainable_parameters()
    
    return model, tokenizer


def setup_wandb(args):
    """
    Setup wandb logging - DISABLED
    """
    logger.info("wandb logging disabled")
    return False


def setup_training_config(args):
    """
    Setup training configuration
    
    Args:
        args: Training arguments
    
    Returns:
        TrainingArguments: Training configuration
    """
    output_dir = args.output_dir + f"_{args.loss_type}" + f"_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        logging_steps=10,
        save_steps=100,
        # Disable wandb
        report_to="none",
        logging_strategy="steps",
        eval_strategy="no",
        save_strategy="steps",
        remove_unused_columns=False,
    )
    
    return training_args, output_dir


def plot_likelihoods(trainer, output_dir: str):
    """
    Plot three figures side by side: chosen likelihood, rejected likelihood, and margin.
    
    Args:
        trainer: CustomDPOTrainer instance with step_metrics
        output_dir: Directory to save the figure
    """
    if not trainer.step_metrics:
        logger.warning("No step metrics recorded. Skipping visualization.")
        return
    
    steps = [m['step'] for m in trainer.step_metrics]
    chosen_likelihoods = [m['chosen_likelihood'] for m in trainer.step_metrics]
    rejected_likelihoods = [m['rejected_likelihood'] for m in trainer.step_metrics]
    margins = [m['margin'] for m in trainer.step_metrics]
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Plot 1: Chosen likelihood
    axes[0].plot(steps, chosen_likelihoods, 'b-', linewidth=2, label='Chosen')
    axes[0].set_xlabel('Step', fontsize=12)
    axes[0].set_ylabel('Likelihood', fontsize=12)
    axes[0].set_title('Model Likelihood to Chosen', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    
    # Plot 2: Rejected likelihood
    axes[1].plot(steps, rejected_likelihoods, 'r-', linewidth=2, label='Rejected')
    axes[1].set_xlabel('Step', fontsize=12)
    axes[1].set_ylabel('Likelihood', fontsize=12)
    axes[1].set_title('Model Likelihood to Rejected', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    
    # Plot 3: Margin (chosen - rejected)
    axes[2].plot(steps, margins, 'g-', linewidth=2, label='Margin')
    axes[2].axhline(y=0, color='k', linestyle='--', alpha=0.5)
    axes[2].set_xlabel('Step', fontsize=12)
    axes[2].set_ylabel('Margin (Chosen - Rejected)', fontsize=12)
    axes[2].set_title('Likelihood Margin', fontsize=14, fontweight='bold')
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()
    
    plt.tight_layout()
    
    # Save figure
    figure_path = os.path.join(output_dir, 'likelihoods_plot.png')
    plt.savefig(figure_path, dpi=150, bbox_inches='tight')
    logger.info(f"Likelihood plots saved to {figure_path}")
    plt.close()



def get_or_generate_data_path(
    model_path: str,
    alpaca_path: str = "datasets/alpaca_data_with_input_500.jsonl",
    output_dir: str = "datasets/dpo/model_generated",
    batch_size: int = 16,
):
    """
    Get the model-specific data path, generating it if it doesn't exist.
    
    Args:
        model_path: Path to the model
        alpaca_path: Path to alpaca dataset for generation
        output_dir: Directory where generated data is stored
        batch_size: Batch size for generation
    
    Returns:
        str: Path to the data file
    """
    model_name = extract_model_name(model_path)
    data_path = os.path.join(output_dir, f"{model_name}.jsonl")
    
    if os.path.exists(data_path):
        logger.info(f"Found existing model-specific data at {data_path}")
        return data_path
    
    logger.info(f"Model-specific data not found at {data_path}. Generating...")
    generated_path = generate_dpo_data(
        model_path=model_path,
        alpaca_path=alpaca_path,
        output_dir=output_dir,
        batch_size=batch_size,
    )
    logger.info(f"Generated data saved to {generated_path}")
    return generated_path


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="DPO Training with HF DPOTrainer")
    
    # Data arguments
    parser.add_argument("--model_path", type=str, default="Qwen/Qwen3-0.6B")
    parser.add_argument("--data_path", type=str, default=None, help="Override data path (if None, uses model-specific data)")
    parser.add_argument("--alpaca_path", type=str, default="datasets/alpaca_data_with_input_500.jsonl", help="Alpaca dataset path for generation")
    parser.add_argument("--test_data_path", type=str, default="datasets/alpaca_data_with_input_test.jsonl")
    parser.add_argument("--num_samples", type=int, default=100)
    parser.add_argument("--gen_batch_size", type=int, default=16, help="Batch size for data generation")
    
    # Training arguments
    parser.add_argument("--output_dir", type=str, default="model_outputs/dpo_qwen3_0.6b")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--loss_type", type=str, default="dpo", help="Loss type: dpo, ipo, tdpo, bdpo, simpo, repo")
    
    # LoRA arguments
    parser.add_argument("--use_lora", action="store_true", default=True, help="Use LoRA for efficient fine-tuning")
    parser.add_argument("--no_lora", action="store_true", help="Disable LoRA (train full model)")
    parser.add_argument("--lora_r", type=int, default=8, help="LoRA rank (lower = fewer parameters)")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha (scaling factor)")
    parser.add_argument("--lora_dropout", type=float, default=0.1, help="LoRA dropout rate")
    parser.add_argument("--lora_target_modules", type=str, nargs="+", default=None, 
                        help="Target modules for LoRA (default: q_proj v_proj)")
    parser.add_argument("--lora_bias", type=str, default="none", choices=["none", "all", "lora_only"],
                        help="Bias handling: none (default), all, or lora_only")
    
    return parser.parse_args()

def train():
    """Main training function"""
    setup()
    args = parse_arguments()
    
    use_lora = args.use_lora and not args.no_lora
    
    # Get or generate model-specific data path
    if args.data_path is None:
        logger.info("No data path specified, using model-specific data...")
        data_path = get_or_generate_data_path(
            model_path=args.model_path,
            alpaca_path=args.alpaca_path,
            batch_size=args.gen_batch_size,
        )
    else:
        logger.info(f"Using specified data path: {args.data_path}")
        data_path = args.data_path
    
    # Setup model and tokenizer
    logger.info("Setting up model and tokenizer...")
    model, tokenizer = setup_model_and_tokenizer(
        model_path=args.model_path,
        use_lora=use_lora,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        lora_target_modules=args.lora_target_modules,
        lora_bias=args.lora_bias
    )
    
    logger.info(f"Model and tokenizer loaded from {args.model_path}")
    
    # Create reference model
    if args.loss_type in ['dpo', 'ipo', 'tdpo', 'bdpo']:
        logger.info("Creating reference model...")
        ref_model = create_ref_model(args.model_path)
        logger.info(f"Ref model loaded from {args.model_path}")
    else:
        logger.info("No reference model needed for this loss type")
        ref_model = None
        
        
    # Load dataset with custom DPODataset
    logger.info(f"Loading custom DPO dataset from {data_path}...")
    dataset = DPODataset(
        data_path=data_path,
        tokenizer=tokenizer,
        num_samples=args.num_samples,
        max_length=256,
    )
    logger.info(f"Loaded {len(dataset)} examples with prompt injection")
    
    # Setup training config
    training_args, output_dir = setup_training_config(args)
    
    # Use simple collate function
    data_collator = dpo_collate_fn
    
    # Initialize custom trainer
    logger.info("Initializing custom DPO trainer...")
    trainer = CustomDPOTrainer(
        model=model,
        ref_model=ref_model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
        loss_fn=args.loss_type,
        beta=0.5,
        return_outputs=True
    )
    # Train
    logger.info("Starting training...")
    trainer.train()
    trainer.save_model()
    logger.info(f"Training completed! Model saved to {output_dir}")
    
    # Plot likelihoods
    logger.info("Generating likelihood plots...")
    plot_likelihoods(trainer, output_dir)
    
    


if __name__ == "__main__":
    args = parse_arguments()
    train()