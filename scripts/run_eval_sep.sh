#!/bin/bash
# Run SEP (instruction-data separation) evaluation
# Usage: ./scripts/run_eval_sep.sh [MODEL_PATH] [DATASET_PATH] [NUM_SAMPLES]

set -euo pipefail

MODEL_PATH="${1:-Qwen/Qwen3-0.6B}"
DATASET_PATH="${2:-datasets/sep_dataset/sep_probe_examples.json}"
NUM_SAMPLES="${3:-208}"

echo "Running SEP evaluation..."
echo "Model: $MODEL_PATH"
echo "Dataset: $DATASET_PATH"
echo "Samples: ${NUM_SAMPLES}"
echo ""

python -m eval.sep_cli \
  --model_path "$MODEL_PATH" \
  --dataset_path "$DATASET_PATH" \
  --num_samples "$NUM_SAMPLES"

