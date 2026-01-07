#!/bin/bash
# Run all three evaluation methods sequentially
# Usage: ./run_all_evals.sh [MODEL_PATH] [NUM_SAMPLES]

MODEL_PATH="${1:-Qwen/Qwen3-0.6B}"
NUM_SAMPLES="${2:-10}"

echo "=========================================="
echo "Running all evaluations for: $MODEL_PATH"
echo "=========================================="
echo ""

echo "1. Running ASR evaluation..."
./scripts/run_eval_asr.sh "$MODEL_PATH" "" "$NUM_SAMPLES"
echo ""

echo "2. Running AlpacaEval (generate only, no scoring)..."
./scripts/run_eval_alpaca.sh "$MODEL_PATH" "" "$NUM_SAMPLES" "true"
echo ""

echo "3. Running InjecAgent evaluation..."
./scripts/run_eval_injecagent.sh "$MODEL_PATH" "base" "InjecAgent" "$NUM_SAMPLES" "true"
echo ""

echo "=========================================="
echo "All evaluations completed!"
echo "=========================================="

