#!/bin/bash

# DPO Training Script for Qwen3 0.6B
# Trains the model to be robust against prompt injection attacks

echo "Starting DPO Training for Qwen3 0.6B..."
echo "Dataset: data/dpo_alpaca_prompt_injection_500.jsonl"
echo "Output: ../outputs/dpo_qwen3_0.6b"
echo "================================"

# Create output directory
mkdir -p ../outputs

# Run DPO training with lightweight settings
python dpo_training.py \
    --data_path data/dpo_alpaca_prompt_injection_500.jsonl \
    --output_dir ../outputs/dpo_qwen3_0.6b \
    --epochs 2 \
    --batch_size 4 \
    --learning_rate 5e-6 \
    --beta 0.1 \
    --max_length 512

echo "================================"
echo "DPO Training completed!"
echo "Trained model saved to: ../outputs/dpo_qwen3_0.6b/final" 