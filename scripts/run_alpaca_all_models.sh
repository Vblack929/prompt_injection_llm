#!/bin/bash
#
# Run AlpacaEval pipeline on the same 6 base models used in run_dpo_training_multi_model.sh.
#
# Default settings:
# - num_samples=104
# - batch_size=32
# - generate_only=1 (so it runs without OPENAI_API_KEY; set ALPACA_GENERATE_ONLY=0 to score)
# - score_existing=0 (set ALPACA_SCORE_EXISTING=1 to score outputs in datasets/alpaca_baselines)
#
# Usage:
#   bash scripts/run_alpaca_all_models.sh
#
# Env overrides:
#   ALPACA_NUM_SAMPLES=104
#   ALPACA_BATCH_SIZE=32
#   ALPACA_MAX_NEW_TOKENS=512
#   ALPACA_GENERATE_ONLY=1
#   ALPACA_DATASET_PATH=   (empty => official AlpacaEval dataset)
#   ALPACA_OUT_ROOT=model_outputs/alpaca_baselines
#   ALPACA_SCORE_EXISTING=0
#   ALPACA_EXISTING_ROOT=datasets/alpaca_baselines
#   ALPACA_EXISTING_OUTPUTS=alpaca_outputs.json
#   RECORD_PATH=outputs/final_results.txt
#

set -euo pipefail

MODELS=(
  "Qwen/Qwen3-1.7B"
  "Qwen/Qwen3-8B-Base"
  "meta-llama/Llama-3.2-3B-Instruct"
  "meta-llama/Llama-3.1-8B-Instruct"
)

ALPACA_NUM_SAMPLES="${ALPACA_NUM_SAMPLES:-104}"
ALPACA_BATCH_SIZE="${ALPACA_BATCH_SIZE:-32}"
ALPACA_MAX_NEW_TOKENS="${ALPACA_MAX_NEW_TOKENS:-512}"
ALPACA_GENERATE_ONLY="${ALPACA_GENERATE_ONLY:-1}"
ALPACA_DATASET_PATH="${ALPACA_DATASET_PATH:-}"
ALPACA_OUT_ROOT="${ALPACA_OUT_ROOT:-model_outputs/alpaca_baselines}"
ALPACA_SCORE_EXISTING="${ALPACA_SCORE_EXISTING:-0}"
ALPACA_EXISTING_ROOT="${ALPACA_EXISTING_ROOT:-datasets/alpaca_baselines}"
ALPACA_EXISTING_OUTPUTS="${ALPACA_EXISTING_OUTPUTS:-alpaca_outputs.json}"
RECORD_PATH="${RECORD_PATH:-outputs/final_results.txt}"

echo "=========================================="
echo "AlpacaEval on base models"
echo "num_samples: $ALPACA_NUM_SAMPLES"
echo "batch_size: $ALPACA_BATCH_SIZE"
echo "max_new_tokens: $ALPACA_MAX_NEW_TOKENS"
echo "generate_only: $ALPACA_GENERATE_ONLY"
echo "dataset_path: ${ALPACA_DATASET_PATH:-official}"
echo "out_root: $ALPACA_OUT_ROOT"
echo "score_existing: $ALPACA_SCORE_EXISTING"
echo "existing_root: $ALPACA_EXISTING_ROOT"
echo "existing_outputs: $ALPACA_EXISTING_OUTPUTS"
echo "record_path: $RECORD_PATH"
echo "=========================================="

if [ "$ALPACA_SCORE_EXISTING" = "1" ]; then
  echo ""
  echo "Scoring existing outputs under: $ALPACA_EXISTING_ROOT"
  TOTAL=$(find "$ALPACA_EXISTING_ROOT" -mindepth 1 -maxdepth 1 -type d | wc -l | tr -d ' ')
  IDX=0
  for DIR in "$ALPACA_EXISTING_ROOT"/*; do
    [ -d "$DIR" ] || continue
    OUTPUTS_PATH="$DIR/$ALPACA_EXISTING_OUTPUTS"
    if [ ! -f "$OUTPUTS_PATH" ]; then
      echo "Skipping: $DIR (missing $ALPACA_EXISTING_OUTPUTS)"
      continue
    fi
    IDX=$((IDX + 1))
    MODEL_TAG="$(basename "$DIR")"
    OUT_DIR="${DIR}/alpaca_eval"
    mkdir -p "$OUT_DIR"
    echo ""
    echo "[$IDX/$TOTAL] $MODEL_TAG"

    python -m eval.alpaca_eval_cli \
      --model_path "$MODEL_TAG" \
      --model_outputs_path "$OUTPUTS_PATH" \
      --output_dir "$OUT_DIR" \
      --record_path "$RECORD_PATH" > "${OUT_DIR}/stdout.json"
  done
else
  TOTAL=${#MODELS[@]}
  IDX=0
  for MODEL in "${MODELS[@]}"; do
    IDX=$((IDX + 1))
    MODEL_TAG="$(echo "$MODEL" | sed 's#/#_#g')"
    OUT_DIR="${ALPACA_OUT_ROOT}/${MODEL_TAG}"
    mkdir -p "$OUT_DIR"
    echo ""
    echo "[$IDX/$TOTAL] $MODEL"

    ARGS=(
      --model_path "$MODEL"
      --num_samples "$ALPACA_NUM_SAMPLES"
      --batch_size "$ALPACA_BATCH_SIZE"
      --max_new_tokens "$ALPACA_MAX_NEW_TOKENS"
      --outputs_path "${OUT_DIR}/alpaca_outputs.json"
      --output_dir "${OUT_DIR}/alpaca_eval"
      --record_path "$RECORD_PATH"
    )

    if [ -n "$ALPACA_DATASET_PATH" ]; then
      ARGS+=(--dataset_path "$ALPACA_DATASET_PATH")
    fi

    if [ "$ALPACA_GENERATE_ONLY" = "1" ]; then
      ARGS+=(--generate_only)
    fi

    python -m eval.alpaca_eval_cli "${ARGS[@]}" > "${OUT_DIR}/stdout.json"
  done
fi

echo ""
echo "Done. Results under: $ALPACA_OUT_ROOT"

