#!/bin/bash
#
# Score existing AlpacaEval outputs JSON (no generation).
#
# Usage:
#   bash scripts/run_alpaca_score_outputs.sh /path/to/outputs.json [MODEL_NAME] [OUTPUT_DIR]
#
# Env overrides:
#   ALPACA_ANNOTATORS_CONFIG=weighted_alpaca_eval_gpt4_turbo
#   ALPACA_VALIDATE_DATASET=1
#
# Notes:
# - Requires OPENAI_API_KEY in the environment.
# - This script never loads a local model or generates responses.
#

set -euo pipefail

OUTPUTS_PATH="${1:-}"
MODEL_NAME="${2:-generated_outputs}"
OUTPUT_DIR="${3:-}"
ALPACA_ANNOTATORS_CONFIG="${ALPACA_ANNOTATORS_CONFIG:-weighted_alpaca_eval_gpt4_turbo}"
ALPACA_VALIDATE_DATASET="${ALPACA_VALIDATE_DATASET:-1}"

if [ -z "$OUTPUTS_PATH" ]; then
  echo "Error: outputs path is required."
  echo "Usage: bash scripts/run_alpaca_score_outputs.sh /path/to/outputs.json [MODEL_NAME] [OUTPUT_DIR]"
  exit 1
fi

if [ ! -f "$OUTPUTS_PATH" ]; then
  echo "Error: outputs file not found: $OUTPUTS_PATH"
  exit 1
fi

if [ -z "${OPENAI_API_KEY:-}" ]; then
  echo "Error: OPENAI_API_KEY is not set."
  exit 1
fi

if [ "$ALPACA_VALIDATE_DATASET" = "1" ]; then
  python - <<'PY'
import json
import sys

outputs_path = sys.argv[1]
try:
    import alpaca_eval  # type: ignore
except Exception:
    sys.exit(0)

try:
    with open(outputs_path, "r", encoding="utf-8") as f:
        outputs = json.load(f)
except Exception as exc:
    print(f"Error: failed to read outputs JSON: {exc}")
    sys.exit(1)

def key(row):
    return (row.get("instruction", "").strip(), row.get("input", "").strip())

out_keys = {key(row) for row in outputs if isinstance(row, dict)}

try:
    data = alpaca_eval.get_data_alpaca_eval()
except Exception:
    sys.exit(0)

data_keys = {(row.get("instruction", "").strip(), row.get("input", "").strip()) for row in data}
overlap = len(out_keys & data_keys)

if overlap == 0:
    print("Error: outputs do not overlap the official AlpacaEval dataset.")
    print("Regenerate outputs using the official AlpacaEval dataset or use a matching annotators config.")
    sys.exit(1)
PY
  "$OUTPUTS_PATH"
fi

ARGS=(
  --model_path "$MODEL_NAME"
  --model_outputs_path "$OUTPUTS_PATH"
  --annotators_config "$ALPACA_ANNOTATORS_CONFIG"
)

if [ -n "$OUTPUT_DIR" ]; then
  ARGS+=(--output_dir "$OUTPUT_DIR")
fi

python -m eval.alpaca_eval_cli "${ARGS[@]}"
