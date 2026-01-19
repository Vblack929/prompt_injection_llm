#!/bin/bash
#
# Train baseline losses (DPO, SimPO, SimPER) on two 8B models with identical training steps.
#
# Keeps training steps identical across runs by fixing:
# - num_samples, epochs, batch_size, gen_batch_size
#
# Usage:
#   bash scripts/run_train_8b_baselines.sh
#

set -euo pipefail

MODELS=(
  "Qwen/Qwen3-8B-Base"
  "meta-llama/Llama-3.1-8B-Instruct"
)

LOSSES=("dpo" "simpo" "simper")

# Fixed training schedule (same across all runs)
EPOCHS="${EPOCHS:-3}"
NUM_SAMPLES="${NUM_SAMPLES:-500}"
BATCH_SIZE="${BATCH_SIZE:-1}"
GEN_BATCH_SIZE="${GEN_BATCH_SIZE:-2}"

# LoRA (fixed)
LORA_R="${LORA_R:-8}"
LORA_ALPHA="${LORA_ALPHA:-32}"
LORA_DROPOUT="${LORA_DROPOUT:-0.1}"
LORA_BIAS="${LORA_BIAS:-none}"

# Learning rates to run (space-separated). Keep identical across losses if you want aligned comparisons.
LR_LIST="${LR_LIST:-5e-5}"

# Loss hparams (kept fixed unless you override via env)
BETA="${BETA:-0.5}"
GAMMA="${GAMMA:-0.5}"
ALPHA="${ALPHA:-0.5}"
LAMBDA_MIX="${LAMBDA_MIX:-0.5}"

OUT_ROOT="${OUT_ROOT:-model_outputs/exp_8b_baselines}"

echo "=========================================="
echo "8B baselines: ${LOSSES[*]}"
echo "models: ${MODELS[*]}"
echo "fixed: epochs=$EPOCHS num_samples=$NUM_SAMPLES batch_size=$BATCH_SIZE gen_batch=$GEN_BATCH_SIZE"
echo "lr_list: $LR_LIST"
echo "out_root: $OUT_ROOT"
echo "=========================================="

for MODEL in "${MODELS[@]}"; do
  MODEL_TAG="$(echo "$MODEL" | sed 's#/#_#g')"
  for LOSS in "${LOSSES[@]}"; do
    for LR in $LR_LIST; do
      RUN_TAG="${MODEL_TAG}__${LOSS}__lr${LR}"
      OUT_DIR="${OUT_ROOT}/${RUN_TAG}"
      mkdir -p "$OUT_DIR"
      echo ""
      echo "[TRAIN] $RUN_TAG"

      python dpo_training.py \
        --model_path "$MODEL" \
        --loss_type "$LOSS" \
        --num_samples "$NUM_SAMPLES" \
        --epochs "$EPOCHS" \
        --batch_size "$BATCH_SIZE" \
        --learning_rate "$LR" \
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
        --output_dir "$OUT_DIR"
    done
  done
done

echo ""
echo "Done. Outputs under: $OUT_ROOT"

