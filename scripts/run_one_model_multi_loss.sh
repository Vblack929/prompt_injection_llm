#!/bin/bash
#
# Run multiple losses (DPO, SimPO, and our behavioral_hard_simper_loss) on ONE model.
# Trains + runs ASR eval for each loss via run_dpo_with_asr.py.
#
# Usage:
#   bash scripts/run_one_model_multi_loss.sh "Qwen/Qwen3-1.7B"
#   bash scripts/run_one_model_multi_loss.sh "meta-llama/Llama-3.2-3B-Instruct"
#
# Notes:
# - DPO loads a reference model (extra memory). For 8B models, this may OOM on 1 GPU.
#   By default, we SKIP dpo on 8B models unless you set FORCE_DPO_8B=1.
# - You can override any hyperparameter by exporting env vars (see defaults below).
#

set -euo pipefail

MODEL="${1:-Qwen/Qwen3-0.6B}"

LOSSES=(
  "dpo"
  "simpo"
  "behavioral_hard_simper_loss"
)

# Shared defaults (override by exporting env vars before running)
EPOCHS="${EPOCHS:-3}"
NUM_SAMPLES="${NUM_SAMPLES:-500}"
LEARNING_RATE="${LEARNING_RATE:-5e-5}"

# Loss hparams
BETA="${BETA:-0.5}"
GAMMA="${GAMMA:-0.5}"
ALPHA="${ALPHA:-0.5}"
LAMBDA_MIX="${LAMBDA_MIX:-0.5}"

# LoRA
LORA_R="${LORA_R:-8}"
LORA_ALPHA="${LORA_ALPHA:-32}"
LORA_DROPOUT="${LORA_DROPOUT:-0.1}"
LORA_BIAS="${LORA_BIAS:-none}"

# ASR eval
EVAL_NUM_SAMPLES="${EVAL_NUM_SAMPLES:-100}"
EVAL_MAX_NEW_TOKENS="${EVAL_MAX_NEW_TOKENS:-512}"

# Basic per-model sizing (override with env vars if needed)
BATCH_SIZE="${BATCH_SIZE:-4}"
GEN_BATCH_SIZE="${GEN_BATCH_SIZE:-16}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-8}"

if [[ "$MODEL" == *"1.7B"* ]]; then
  BATCH_SIZE="${BATCH_SIZE:-2}"
  GEN_BATCH_SIZE="${GEN_BATCH_SIZE:-8}"
  EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-4}"
elif [[ "$MODEL" == *"3B"* ]]; then
  BATCH_SIZE="${BATCH_SIZE:-1}"
  GEN_BATCH_SIZE="${GEN_BATCH_SIZE:-4}"
  EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-2}"
elif [[ "$MODEL" == *"8B"* ]]; then
  BATCH_SIZE="${BATCH_SIZE:-1}"
  GEN_BATCH_SIZE="${GEN_BATCH_SIZE:-2}"
  EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-1}"
fi

echo "=========================================="
echo "One-model multi-loss run: train + ASR eval"
echo "Model: $MODEL"
echo "Losses: ${LOSSES[*]}"
echo "train_batch=$BATCH_SIZE  gen_batch=$GEN_BATCH_SIZE  eval_batch=$EVAL_BATCH_SIZE"
echo "=========================================="

TOTAL=${#LOSSES[@]}
CURRENT=0

for LOSS in "${LOSSES[@]}"; do
  CURRENT=$((CURRENT + 1))

  if [[ "$LOSS" == "dpo" && "$MODEL" == *"8B"* && "${FORCE_DPO_8B:-0}" != "1" ]]; then
    echo ""
    echo "[$CURRENT/$TOTAL] SKIP: $LOSS on $MODEL (likely OOM due to ref model)."
    echo "Set FORCE_DPO_8B=1 to force running it anyway."
    continue
  fi

  echo ""
  echo "=========================================="
  echo "[$CURRENT/$TOTAL] Loss: $LOSS"
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
echo "Done."
echo "Each run writes: <model_output_dir>/asr_eval/asr_results.json and summary.txt"
echo "=========================================="

