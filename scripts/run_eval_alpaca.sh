#!/bin/bash
# Run AlpacaEval 2.0 evaluation
# Usage: ./run_eval_alpaca.sh [MODEL_PATH] [DATASET_PATH] [NUM_SAMPLES] [GENERATE_ONLY] [MODEL_OUTPUTS_PATH]

MODEL_PATH="${1:-Qwen/Qwen3-0.6B}"
DATASET_PATH="${2:-}"
NUM_SAMPLES="${3:-}"
GENERATE_ONLY="${4:-false}"
MODEL_OUTPUTS_PATH="${5:-}"

echo "Running AlpacaEval evaluation..."
echo "Model: $MODEL_PATH"
echo "Dataset: ${DATASET_PATH:-default AlpacaEval}"
echo "Samples: ${NUM_SAMPLES:-all}"
echo "Generate only: $GENERATE_ONLY"
echo "Model outputs: ${MODEL_OUTPUTS_PATH:-none}"
echo ""

if [ -n "$MODEL_OUTPUTS_PATH" ]; then
    if [ "$GENERATE_ONLY" = "true" ]; then
        echo "Error: GENERATE_ONLY=true is incompatible with MODEL_OUTPUTS_PATH"
        exit 1
    fi
    python -m eval.alpaca_eval_cli \
        --model_path "$MODEL_PATH" \
        --model_outputs_path "$MODEL_OUTPUTS_PATH"
else
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
fi

