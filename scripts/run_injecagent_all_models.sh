#!/bin/bash
#
# Run InjecAgent evaluation on the same 6 base models used in run_dpo_training_multi_model.sh.
# Defaults: num_samples=104, batch_size=32.
#
# Usage:
#   bash scripts/run_injecagent_all_models.sh
#
# Env overrides:
#   INJ_NUM_SAMPLES=104
#   INJ_BATCH_SIZE=32
#   INJ_SETTING=base
#   INJ_PROMPT_TYPE=InjecAgent
#   INJ_ONLY_FIRST_STEP=1
#   INJ_OUT_DIR=outputs/injecagent
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

INJ_NUM_SAMPLES="${INJ_NUM_SAMPLES:-104}"
INJ_BATCH_SIZE="${INJ_BATCH_SIZE:-32}"
INJ_SETTING="${INJ_SETTING:-base}"
INJ_PROMPT_TYPE="${INJ_PROMPT_TYPE:-InjecAgent}"
INJ_ONLY_FIRST_STEP="${INJ_ONLY_FIRST_STEP:-1}"
INJ_OUT_DIR="${INJ_OUT_DIR:-outputs/injecagent}"
RECORD_PATH="${RECORD_PATH:-outputs/final_results.txt}"

echo "=========================================="
echo "InjecAgent eval on base models"
echo "setting: $INJ_SETTING"
echo "prompt_type: $INJ_PROMPT_TYPE"
echo "num_samples: $INJ_NUM_SAMPLES"
echo "batch_size: $INJ_BATCH_SIZE"
echo "only_first_step: $INJ_ONLY_FIRST_STEP"
echo "out_dir: $INJ_OUT_DIR"
echo "record_path: $RECORD_PATH"
echo "=========================================="

TOTAL=${#MODELS[@]}
IDX=0
for MODEL in "${MODELS[@]}"; do
  IDX=$((IDX + 1))
  echo ""
  echo "[$IDX/$TOTAL] $MODEL"

  EXTRA_ARGS=()
  if [ "$INJ_ONLY_FIRST_STEP" = "1" ]; then
    EXTRA_ARGS+=(--only_first_step)
  fi

  python -m eval.injecagent_cli \
    --model_path "$MODEL" \
    --setting "$INJ_SETTING" \
    --prompt_type "$INJ_PROMPT_TYPE" \
    --num_samples "$INJ_NUM_SAMPLES" \
    --batch_size "$INJ_BATCH_SIZE" \
    --out_dir "$INJ_OUT_DIR" \
    --record_path "$RECORD_PATH" \
    "${EXTRA_ARGS[@]}"
done

echo ""
echo "Done. Outputs under: $INJ_OUT_DIR"

