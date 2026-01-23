#!/bin/bash
#
# Sweep hyperparameters for our method (BHPO) on 8B models.
#
# Only changes:
# - learning_rate
# - alpha  (mapped to sharpness in BHPO)
# - lambda_mix (mapped to lambda_anchor in BHPO; see loss.py mapping)
#
# Keeps training steps identical across runs by fixing:
# - num_samples, epochs, batch_size, gen_batch_size
# and using the same cached model-specific dataset file.
#
# Usage:
#   bash scripts/run_train_8b_our_method_sweep.sh
#

set -euo pipefail

LOSS="bhpo"

MODELS=(
  "Qwen/Qwen3-8B-Base"
  "meta-llama/Llama-3.1-8B-Instruct"
)

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

# Sweep knobs (space-separated lists)
LR_LIST="${LR_LIST:-5e-5 3e-5 1e-5}"
ALPHA_LIST="${ALPHA_LIST:1.0}"
LAMBDA_ANCHOR_LIST="${LAMBDA_ANCHOR_LIST:-0.02 0.05}"

OUT_ROOT="${OUT_ROOT:-model_outputs/exp_8b_our_method}"

echo "=========================================="
echo "8B sweep: $LOSS"
echo "models: ${MODELS[*]}"
echo "fixed: epochs=$EPOCHS num_samples=$NUM_SAMPLES batch_size=$BATCH_SIZE gen_batch=$GEN_BATCH_SIZE"
echo "lr_list: $LR_LIST"
echo "alpha_list: $ALPHA_LIST"
echo "lambda_anchor_list: $LAMBDA_ANCHOR_LIST"
echo "out_root: $OUT_ROOT"
echo "=========================================="

for MODEL in "${MODELS[@]}"; do
  MODEL_TAG="$(echo "$MODEL" | sed 's#/#_#g')"
  for LR in $LR_LIST; do
    for ALPHA in $ALPHA_LIST; do
      for LAM in $LAMBDA_ANCHOR_LIST; do
        RUN_TAG="${MODEL_TAG}__${LOSS}__lr${LR}__a${ALPHA}__lam${LAM}"
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
          --alpha "$ALPHA" \
          --lambda_mix "$LAM" \
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
done

echo ""
echo "Done. Outputs under: $OUT_ROOT"

