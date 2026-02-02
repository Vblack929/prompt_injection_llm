#!/bin/bash
#
# Ablation study: BHPO training with fixed LR and different lambda values
# Runs training + ASR eval for each lambda value
#
# Usage:
#   bash scripts/ablation.sh
#

set -euo pipefail

LOSS="bhpo"

# Model to use (can be overridden via MODEL env var)
MODEL="${MODEL:-meta-llama/Llama-3.1-8B-Instruct}"

# Fixed training parameters
EPOCHS="${EPOCHS:-3}"
NUM_SAMPLES="${NUM_SAMPLES:-500}"
BATCH_SIZE="${BATCH_SIZE:-8}"
GEN_BATCH_SIZE="${GEN_BATCH_SIZE:-8}"
LEARNING_RATE="${LEARNING_RATE:-5e-5}"  # Fixed learning rate

# Fixed BHPO parameters
ALPHA="${ALPHA:-1.0}"  # Fixed alpha (sharpness)

# Lambda values to sweep (lambda_mix / lambda_anchor)
LAMBDA_VALUES=(0.0 0.01 0.05 0.1)

# LoRA (fixed)
LORA_R="${LORA_R:-8}"
LORA_ALPHA="${LORA_ALPHA:-32}"
LORA_DROPOUT="${LORA_DROPOUT:-0.1}"
LORA_BIAS="${LORA_BIAS:-none}"

# ASR eval config
EVAL_NUM_SAMPLES="${EVAL_NUM_SAMPLES:-104}"
EVAL_MAX_NEW_TOKENS="${EVAL_MAX_NEW_TOKENS:-512}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-16}"

# Output root
OUT_ROOT="${OUT_ROOT:-model_outputs/ablation_bhpo_lambda}"

MODEL_TAG="$(echo "$MODEL" | sed 's#/#_#g')"

echo "=========================================="
echo "Ablation Study: BHPO Lambda Sweep"
echo "=========================================="
echo "Model: $MODEL"
echo "Loss: $LOSS"
echo "Fixed LR: $LEARNING_RATE"
echo "Fixed Alpha: $ALPHA"
echo "Lambda values: ${LAMBDA_VALUES[*]}"
echo "Training: epochs=$EPOCHS num_samples=$NUM_SAMPLES batch_size=$BATCH_SIZE"
echo "ASR Eval: num_samples=$EVAL_NUM_SAMPLES max_tokens=$EVAL_MAX_NEW_TOKENS"
echo "Output root: $OUT_ROOT"
echo "=========================================="

TOTAL=${#LAMBDA_VALUES[@]}
CURRENT=0

for LAM in "${LAMBDA_VALUES[@]}"; do
  CURRENT=$((CURRENT + 1))
  
  RUN_TAG="${MODEL_TAG}__${LOSS}__lr${LEARNING_RATE}__a${ALPHA}__lam${LAM}"
  OUT_DIR="${OUT_ROOT}/${RUN_TAG}"
  
  echo ""
  echo "[$CURRENT/$TOTAL] Training with lambda=$LAM"
  echo "Output: $OUT_DIR"
  echo "----------------------------------------"
  
  # Run training + ASR eval using run_dpo_with_asr.py
  python run_dpo_with_asr.py \
    --model_path "$MODEL" \
    --loss_type "$LOSS" \
    --num_samples "$NUM_SAMPLES" \
    --epochs "$EPOCHS" \
    --batch_size "$BATCH_SIZE" \
    --learning_rate "$LEARNING_RATE" \
    --alpha "$ALPHA" \
    --lambda_mix "$LAM" \
    --use_lora \
    --lora_r "$LORA_R" \
    --lora_alpha "$LORA_ALPHA" \
    --lora_dropout "$LORA_DROPOUT" \
    --lora_bias "$LORA_BIAS" \
    --gen_batch_size "$GEN_BATCH_SIZE" \
    --output_dir "$OUT_DIR" \
    --eval_num_samples "$EVAL_NUM_SAMPLES" \
    --eval_max_new_tokens "$EVAL_MAX_NEW_TOKENS" \
    --eval_batch_size "$EVAL_BATCH_SIZE"
  
  echo ""
  echo "Completed lambda=$LAM"
  echo "Results: $OUT_DIR/asr_eval/"
done

echo ""
echo "=========================================="
echo "Ablation study complete!"
echo "=========================================="
echo "Trained and evaluated ${TOTAL} models with lambda values: ${LAMBDA_VALUES[*]}"
echo "Results saved under: $OUT_ROOT"
echo ""
echo "ASR results summary:"
for LAM in "${LAMBDA_VALUES[@]}"; do
  RUN_TAG="${MODEL_TAG}__${LOSS}__lr${LEARNING_RATE}__a${ALPHA}__lam${LAM}"
  OUT_DIR="${OUT_ROOT}/${RUN_TAG}"
  SUMMARY_FILE="${OUT_DIR}/asr_eval/summary.txt"
  if [ -f "$SUMMARY_FILE" ]; then
    echo "  Lambda $LAM: $SUMMARY_FILE"
  fi
done
echo "=========================================="
