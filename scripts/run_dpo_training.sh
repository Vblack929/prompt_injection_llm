#!/bin/bash

# DPO training with custom loss functions
# Usage: ./run_dpo_training.sh [loss_type]
# Available loss types: dpo, ipo, tdpo, bdpo, simpo, repo, simper
# 
# Loss-specific hyperparameters:
#   - DPO/IPO/TDPO/BDPO: beta (default: 0.5)
#   - TDPO: alpha (default: 0.5)
#   - BDPO: lambda_mix (default: 0.5)
#   - SimPO: beta (default: 2.0), gamma (default: 0.5)
#   - RePO: gamma (default: 0.5)
# 
# Note: Data is automatically generated if model-specific data doesn't exist.
# To use a specific data file, add --data_path "path/to/data.jsonl"

LOSS_TYPE=${1:-"dpo"}
echo "Starting DPO Training with loss: $LOSS_TYPE"

python dpo_training.py \
    --model_path "Qwen/Qwen3-0.6B" \
    --output_dir "model_outputs/dpo_qwen3_0.6b" \
    --epochs 3 \
    --batch_size 4 \
    --learning_rate 5e-5 \
    --loss_type "$LOSS_TYPE" \
    --num_samples 500 \
    --beta 0.5 \
    --gamma 0.5 \
    --alpha 0.5 \
    --lambda_mix 0.5 \
    --use_lora \
    --lora_r 8 \
    --lora_alpha 32 \
    --lora_dropout 0.1 \
    --lora_bias none \
    --gen_batch_size 16

echo "DPO Training completed!"
