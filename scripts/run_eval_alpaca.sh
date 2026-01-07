#!/bin/bash
# Run AlpacaEval 2.0 evaluation
# Usage: ./run_eval_alpaca.sh [MODEL_PATH] [DATASET_PATH] [NUM_SAMPLES] [GENERATE_ONLY]

MODEL_PATH="${1:-Qwen/Qwen3-0.6B}"
DATASET_PATH="${2:-}"
NUM_SAMPLES="${3:-}"
GENERATE_ONLY="${4:-false}"

echo "Running AlpacaEval evaluation..."
echo "Model: $MODEL_PATH"
echo "Dataset: ${DATASET_PATH:-default AlpacaEval}"
echo "Samples: ${NUM_SAMPLES:-all}"
echo "Generate only: $GENERATE_ONLY"
echo ""

if [ "$GENERATE_ONLY" = "true" ]; then
    python -m eval.alpaca_eval_cli \
        --model_path "$MODEL_PATH" \
        --dataset_path "$DATASET_PATH" \
        --num_samples "$NUM_SAMPLES" \
        --generate_only
else
    python -m eval.alpaca_eval_cli \
        --model_path "$MODEL_PATH" \
        --dataset_path "$DATASET_PATH" \
        --num_samples "$NUM_SAMPLES"
fi

