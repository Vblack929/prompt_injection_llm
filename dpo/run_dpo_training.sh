#!/bin/bash

# DPO Training Script for Qwen3 0.6B
# Trains the model to be robust against prompt injection attacks

echo "Starting DPO Training for Qwen3 0.6B with LoRA..."
echo "Dataset: data/dpo_alpaca_prompt_injection_500.jsonl"
echo "Output: ../outputs/dpo_qwen3_0.6b_lora"
echo "Using LoRA for memory-efficient training"
echo "================================"

# Create output directory

DATA_PATH=data/dpo_data_train.jsonl
OUTPUT_DIR=model_outputs/dpo_qwen3_0.6b_lora
EPOCHS=2
BATCH_SIZE=8
LEARNING_RATE=5e-5
LORA_R=16
LORA_ALPHA=32
NUM_SAMPLES=100
mkdir -p model_outputs

# Run DPO training with LoRA (lightweight settings)
python dpo_training.py \
    --data_path $DATA_PATH \
    --output_dir $OUTPUT_DIR \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --learning_rate $LEARNING_RATE \
    --use_lora \
    --lora_r $LORA_R \
    --lora_alpha $LORA_ALPHA \
    --num_samples $NUM_SAMPLES

echo "================================"
echo "DPO Training completed!"
echo "LoRA adapters saved to: model_outputs/dpo_qwen3_0.6b_lora/final" 