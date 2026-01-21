#!/bin/bash
#
# Convenience wrapper: run AlpacaEval, SEP, and InjecAgent on all base models.
# Ensures all three pipelines append their summary records to ONE txt file.
#
# Usage:
#   bash scripts/run_all_base_model_evals.sh
#
# Env overrides:
#   RECORD_PATH=outputs/final_results.txt
#   ALPACA_SCORE_EXISTING=0|1
#   ALPACA_EXISTING_ROOT=datasets/alpaca_baselines
#   ALPACA_EXISTING_OUTPUTS=alpaca_outputs.json
#   (plus all env vars supported by the underlying scripts)
#

set -euo pipefail

RECORD_PATH="${RECORD_PATH:-outputs/final_results.txt}"
export RECORD_PATH

echo "=========================================="
echo "Run all base-model evals: AlpacaEval + SEP + InjecAgent"
echo "record_path: $RECORD_PATH"
echo "=========================================="

echo ""
if [ "${ALPACA_SCORE_EXISTING:-0}" = "1" ]; then
  echo "== AlpacaEval (scoring existing outputs only; requires OPENAI_API_KEY) =="
else
  echo "== AlpacaEval (default: generate-only unless ALPACA_GENERATE_ONLY=0) =="
fi
bash scripts/run_alpaca_all_models.sh

echo ""
echo "== SEP =="
bash scripts/run_sep_all_models.sh

echo ""
echo "== InjecAgent =="
bash scripts/run_injecagent_all_models.sh

echo ""
echo "=========================================="
echo "Done. Aggregated records appended to: $RECORD_PATH"
echo "=========================================="

