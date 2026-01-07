#!/bin/bash
# Run DPO training followed by ASR evaluation
# Usage: ./run_dpo_training_with_asr_eval.sh [loss_type] [model_path] [num_samples] [eval_num_samples]

LOSS_TYPE=${1:-"dpo"}
MODEL_PATH=${2:-"Qwen/Qwen3-0.6B"}
NUM_SAMPLES=${3:-500}
EVAL_NUM_SAMPLES=${4:-100}

echo "=========================================="
echo "DPO Training + ASR Evaluation Pipeline"
echo "=========================================="
echo "Loss type: $LOSS_TYPE"
echo "Model: $MODEL_PATH"
echo "Training samples: $NUM_SAMPLES"
echo "Eval samples: $EVAL_NUM_SAMPLES"
echo ""

# Run the complete pipeline using Python script
python run_dpo_with_asr.py \
    --model_path "$MODEL_PATH" \
    --loss_type "$LOSS_TYPE" \
    --num_samples "$NUM_SAMPLES" \
    --epochs 3 \
    --batch_size 4 \
    --learning_rate 5e-5 \
    --beta 0.5 \
    --gamma 0.5 \
    --alpha 0.5 \
    --lambda_mix 0.5 \
    --use_lora \
    --lora_r 8 \
    --lora_alpha 32 \
    --lora_dropout 0.1 \
    --lora_bias none \
    --gen_batch_size 16 \
    --eval_num_samples "$EVAL_NUM_SAMPLES"

