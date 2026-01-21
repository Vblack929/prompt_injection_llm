#!/bin/bash
#
# Generate AlpacaEval outputs and score them for all base models.
# Uses the official AlpacaEval dataset only.
#
# Usage:
#   bash scripts/run_alpaca_generate_and_score_all_models.sh
#
# Env overrides:
#   ALPACA_NUM_SAMPLES=104
#   ALPACA_BATCH_SIZE=32
#   ALPACA_MAX_NEW_TOKENS=512
#   ALPACA_ANNOTATORS_CONFIG=weighted_alpaca_eval_gpt4_turbo
#   ALPACA_OUT_ROOT=model_outputs/alpaca_baselines
#   RECORD_PATH=outputs/final_results.txt
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

ALPACA_NUM_SAMPLES="${ALPACA_NUM_SAMPLES:-104}"
ALPACA_BATCH_SIZE="${ALPACA_BATCH_SIZE:-32}"
ALPACA_MAX_NEW_TOKENS="${ALPACA_MAX_NEW_TOKENS:-512}"
ALPACA_ANNOTATORS_CONFIG="${ALPACA_ANNOTATORS_CONFIG:-weighted_alpaca_eval_gpt4_turbo}"
ALPACA_OUT_ROOT="${ALPACA_OUT_ROOT:-model_outputs/alpaca_baselines}"
RECORD_PATH="${RECORD_PATH:-outputs/final_results.txt}"

if [ -z "${OPENAI_API_KEY:-}" ]; then
  echo "Error: OPENAI_API_KEY is not set."
  exit 1
fi

# Require official AlpacaEval dataset to avoid mismatches.
export ALPACA_REQUIRE_OFFICIAL=1

echo "=========================================="
echo "AlpacaEval generate + score (official dataset)"
echo "num_samples: $ALPACA_NUM_SAMPLES"
echo "batch_size: $ALPACA_BATCH_SIZE"
echo "max_new_tokens: $ALPACA_MAX_NEW_TOKENS"
echo "annotators_config: $ALPACA_ANNOTATORS_CONFIG"
echo "out_root: $ALPACA_OUT_ROOT"
echo "record_path: $RECORD_PATH"
echo "=========================================="

TOTAL=${#MODELS[@]}
IDX=0
for MODEL in "${MODELS[@]}"; do
  IDX=$((IDX + 1))
  MODEL_TAG="$(echo "$MODEL" | sed 's#/#_#g')"
  OUT_DIR="${ALPACA_OUT_ROOT}/${MODEL_TAG}"
  mkdir -p "$OUT_DIR"
  echo ""
  echo "[$IDX/$TOTAL] $MODEL"

  python -m eval.alpaca_eval_cli \
    --model_path "$MODEL" \
    --num_samples "$ALPACA_NUM_SAMPLES" \
    --batch_size "$ALPACA_BATCH_SIZE" \
    --max_new_tokens "$ALPACA_MAX_NEW_TOKENS" \
    --outputs_path "${OUT_DIR}/alpaca_outputs.json" \
    --output_dir "${OUT_DIR}/alpaca_eval" \
    --annotators_config "$ALPACA_ANNOTATORS_CONFIG" \
    --record_path "$RECORD_PATH" > "${OUT_DIR}/stdout.json"
done

echo ""
echo "Done. Results under: $ALPACA_OUT_ROOT"
