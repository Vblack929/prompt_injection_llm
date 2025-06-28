#!/bin/bash

# DPO Training Script for Qwen3 0.6B
# Trains the model to be robust against prompt injection attacks

echo "Starting DPO Training for Qwen3 0.6B with LoRA..."
echo "Dataset: data/dpo_alpaca_prompt_injection_500.jsonl"
echo "Output: ../outputs/dpo_qwen3_0.6b_lora"
echo "Using LoRA for memory-efficient training"
echo "================================"

# Create output directory
mkdir -p ../outputs

# Run DPO training with LoRA (lightweight settings)
python dpo_training.py \
    --data_path data/dpo_alpaca_prompt_injection_500.jsonl \
    --output_dir ../outputs/dpo_qwen3_0.6b_lora \
    --epochs 2 \
    --batch_size 8 \
    --learning_rate 5e-5 \
    --use_lora \
    --lora_r 16 \
    --lora_alpha 32 \
    --num_samples 200

echo "================================"
echo "DPO Training completed!"
echo "LoRA adapters saved to: ../outputs/dpo_qwen3_0.6b_lora/final" 