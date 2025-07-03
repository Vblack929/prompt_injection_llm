#!/usr/bin/env python3
"""
Lightweight DPO Training for Qwen3 0.6B Model using HF DPOTrainer
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import LoraConfig, get_peft_model, TaskType
from trl import DPOTrainer, DPOConfig
from datasets import Dataset
import jsonlines
import argparse
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def load_dpo_dataset(data_path: str, num_samples: int = None):
    """Load and format DPO dataset for HF DPOTrainer"""
    data = []
    with jsonlines.open(data_path, 'r') as reader:
        for item in reader:
            data.append({
                'prompt': item['prompt'],
                'chosen': item['chosen'], 
                'rejected': item['rejected']
            })
            if num_samples is not None and len(data) >= num_samples:
                break
    
    return Dataset.from_list(data)

def setup_model_and_tokenizer(model_name: str, use_lora: bool = True, 
                             lora_r: int = 16, lora_alpha: int = 32, lora_dropout: float = 0.1):
    """Setup model and tokenizer with optional LoRA"""
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        trust_remote_code=True
    )
    
    if use_lora:
        # Configure LoRA
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=["q_proj", "v_proj"],
            bias="none",
        )
        model = get_peft_model(model, lora_config)
        
        # Print trainable parameters
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        logger.info(f"Trainable parameters: {trainable_params:,} ({trainable_params/total_params*100:.2f}%)")
    
    return model, tokenizer

def main():
    parser = argparse.ArgumentParser(description="DPO Training with HF DPOTrainer")
    parser.add_argument("--data_path", type=str, default="data/dpo_alpaca_prompt_injection_500.jsonl")
    parser.add_argument("--output_dir", type=str, default="../outputs/dpo_qwen3_0.6b_lora")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--num_samples", type=int, default=None)
    
    # LoRA arguments
    parser.add_argument("--use_lora", action="store_true", default=True)
    parser.add_argument("--no_lora", action="store_true")
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.1)
    
    args = parser.parse_args()
    
    use_lora = args.use_lora and not args.no_lora
    
    # Load dataset
    dataset = load_dpo_dataset(args.data_path, args.num_samples)
    logger.info(f"Loaded {len(dataset)} examples")
    
    # Setup model and tokenizer
    model, tokenizer = setup_model_and_tokenizer(
        "Qwen/Qwen3-0.6B", 
        use_lora=use_lora,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha, 
        lora_dropout=args.lora_dropout
    )
    
    # Training arguments
    # training_args = TrainingArguments(
    #     output_dir=args.output_dir,
    #     num_train_epochs=args.epochs,
    #     per_device_train_batch_size=args.batch_size,
    #     learning_rate=args.learning_rate,
    #     logging_steps=10,
    #     save_steps=100,
    #     eval_steps=100,
    #     warmup_steps=50,
    #     remove_unused_columns=False,
    #     report_to=None,
    # )
    dpo_config = DPOConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        logging_steps=10,
        save_steps=100,
    )
    
    # Initialize DPO trainer
    trainer = DPOTrainer(
        model=model,
        args=dpo_config,
        train_dataset=dataset,
        processing_class=tokenizer,
    )
    
    # Train
    trainer.train()
    
    # Save final model
    trainer.save_model()
    logger.info(f"Training completed! Model saved to {args.output_dir}")

if __name__ == "__main__":
    main() 