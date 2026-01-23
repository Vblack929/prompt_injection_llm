#!/bin/bash

# Final experiment runner (train + ASR eval) for 6 models with BHPO
# Usage:
#   bash scripts/run_dpo_training_multi_model.sh
#
# Note:
# - Uses run_dpo_with_asr.py so each run trains then evaluates ASR and writes a summary file.
# - Uses conservative batch sizes for 8B models to avoid OOM on a single A100 80GB.

set -euo pipefail

LOSS="bhpo"

# 3 Qwen3 + 3 Llama
MODELS=(
  "Qwen/Qwen3-0.6B"
  "Qwen/Qwen3-1.7B"
  "Qwen/Qwen3-8B-Base"
  "meta-llama/Llama-3.2-1B-Instruct"
  "meta-llama/Llama-3.2-3B-Instruct"
  "meta-llama/Llama-3.1-8B-Instruct"
)

# Shared defaults (override per model below)
EPOCHS=3
NUM_SAMPLES=500
LEARNING_RATE="5e-5"

# Loss hparams (BHPO uses alpha; beta/gamma are harmless pass-throughs)
BETA="0.5"
GAMMA="0.5"
ALPHA="0.5"
LAMBDA_MIX="0.5"

# LoRA (keep fixed for paper runs)
LORA_R=8
LORA_ALPHA=32
LORA_DROPOUT="0.1"
LORA_BIAS="none"

# ASR eval config
EVAL_NUM_SAMPLES=100
EVAL_MAX_NEW_TOKENS=512

echo "=========================================="
echo "Final 6-model run: train + ASR eval"
echo "Loss: $LOSS"
echo "Models: ${MODELS[*]}"
echo "=========================================="

TOTAL=${#MODELS[@]}
CURRENT=0

for MODEL in "${MODELS[@]}"; do
  CURRENT=$((CURRENT + 1))

  # Per-model sizing (training batch, generation batch, eval batch)
  BATCH_SIZE=4
  GEN_BATCH_SIZE=16
  EVAL_BATCH_SIZE=8

  case "$MODEL" in
    *"Qwen3-1.7B"*)
      BATCH_SIZE=2
      GEN_BATCH_SIZE=8
      EVAL_BATCH_SIZE=4
      ;;
    *"Qwen3-8B"*|*"Llama-3.1-8B"*)
      BATCH_SIZE=1
      GEN_BATCH_SIZE=2
      EVAL_BATCH_SIZE=1
      ;;
    *"Llama-3.2-3B"*)
      BATCH_SIZE=1
      GEN_BATCH_SIZE=4
      EVAL_BATCH_SIZE=2
      ;;
  esac

  echo ""
  echo "=========================================="
  echo "[$CURRENT/$TOTAL] $MODEL"
  echo "  train_batch=$BATCH_SIZE  gen_batch=$GEN_BATCH_SIZE  eval_batch=$EVAL_BATCH_SIZE"
  echo "=========================================="

  python run_dpo_with_asr.py \
    --model_path "$MODEL" \
    --loss_type "$LOSS" \
    --num_samples "$NUM_SAMPLES" \
    --epochs "$EPOCHS" \
    --batch_size "$BATCH_SIZE" \
    --learning_rate "$LEARNING_RATE" \
    --beta "$BETA" \
    --gamma "$GAMMA" \
    --alpha "$ALPHA" \
    --lambda_mix "$LAMBDA_MIX" \
    --use_lora \
    --lora_r "$LORA_R" \
    --lora_alpha "$LORA_ALPHA" \
    --lora_dropout "$LORA_DROPOUT" \
    --lora_bias "$LORA_BIAS" \
    --gen_batch_size "$GEN_BATCH_SIZE" \
    --eval_num_samples "$EVAL_NUM_SAMPLES" \
    --eval_max_new_tokens "$EVAL_MAX_NEW_TOKENS" \
    --eval_batch_size "$EVAL_BATCH_SIZE"
done

echo ""
echo "=========================================="
echo "Done. Trained + ASR-evaluated ${TOTAL} models."
echo "Each run writes: <model_output_dir>/asr_eval/asr_results.json and summary.txt"
echo "=========================================="

