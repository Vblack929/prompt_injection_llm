#!/bin/bash
#
# Run SEP evaluation on the same 6 base models used in run_dpo_training_multi_model.sh.
# Defaults: num_samples=104, batch_size=32.
#
# Usage:
#   bash scripts/run_sep_all_models.sh
#
# Env overrides:
#   SEP_NUM_SAMPLES=104
#   SEP_BATCH_SIZE=32
#   SEP_DATASET_PATH=datasets/sep_dataset/sep_probe_examples.json
#   SEP_DEFENSES="none sandwich instructional reminder"
#   SEP_OUT_ROOT=model_outputs/sep_baselines
#

set -euo pipefail

MODELS=(
  "Qwen/Qwen3-0.6B"
  "Qwen/Qwen3-1.7B"
  "Qwen/Qwen3-8B-Base"
  "meta-llama/Llama-3.2-1B-Instruct"
  "meta-llama/Llama-3.2-3B-Instruct"
  "meta-llama/Llama-3.1-8B-Instruct"
)

SEP_NUM_SAMPLES="${SEP_NUM_SAMPLES:-104}"
SEP_BATCH_SIZE="${SEP_BATCH_SIZE:-32}"
SEP_DATASET_PATH="${SEP_DATASET_PATH:-datasets/sep_dataset/sep_probe_examples.json}"
SEP_DEFENSES_STR="${SEP_DEFENSES:-none}"
SEP_OUT_ROOT="${SEP_OUT_ROOT:-model_outputs/sep_baselines}"
RECORD_PATH="${RECORD_PATH:-outputs/final_results.txt}"

read -r -a DEFENSES <<< "$SEP_DEFENSES_STR"

echo "=========================================="
echo "SEP eval on base models"
echo "Dataset: $SEP_DATASET_PATH"
echo "num_samples: $SEP_NUM_SAMPLES"
echo "batch_size: $SEP_BATCH_SIZE"
echo "defenses: ${DEFENSES[*]}"
echo "out_root: $SEP_OUT_ROOT"
echo "record_path: $RECORD_PATH"
echo "=========================================="

TOTAL=${#MODELS[@]}
IDX=0
for MODEL in "${MODELS[@]}"; do
  IDX=$((IDX + 1))
  MODEL_TAG="$(echo "$MODEL" | sed 's#/#_#g')"
  echo ""
  echo "[$IDX/$TOTAL] $MODEL"

  for DEF in "${DEFENSES[@]}"; do
    OUT_DIR="${SEP_OUT_ROOT}/${MODEL_TAG}/${DEF}"
    mkdir -p "$OUT_DIR"

    python -m eval.sep_cli \
      --model_path "$MODEL" \
      --dataset_path "$SEP_DATASET_PATH" \
      --num_samples "$SEP_NUM_SAMPLES" \
      --batch_size "$SEP_BATCH_SIZE" \
      --defense "$DEF" \
      --record_path "$RECORD_PATH" \
      --output_path "${OUT_DIR}/sep_results.json" \
      > "${OUT_DIR}/stdout.json"
  done
done

echo ""
echo "Done. Results under: $SEP_OUT_ROOT"

