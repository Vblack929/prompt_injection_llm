#!/bin/bash

# Simple test script using utils.py test_model function
# Usage: ./run_test.sh [model_path] [num_samples]

set -e  # Exit on any error

# Default configuration
DEFAULT_MODEL_PATH="Qwen/Qwen3-0.6B"
DEFAULT_NUM_SAMPLES=100

# Parse arguments
MODEL_PATH=${1:-$DEFAULT_MODEL_PATH}
NUM_SAMPLES=${2:-$DEFAULT_NUM_SAMPLES}

echo "=== DPO Model Prompt Injection Test ==="
echo "Model: $MODEL_PATH"
echo "Samples: $NUM_SAMPLES"
echo "========================================"

# Run the test using utils.py
echo "Starting test..."
python -c "
from utils import test_model
test_model('$MODEL_PATH', num_samples=$NUM_SAMPLES)
"

echo ""
echo "Test completed successfully!"

# Optional: Run multiple models if TEST_MODELS environment variable is set
if [ ! -z "$TEST_MODELS" ]; then
    echo ""
    echo "=== Running tests on multiple models ==="
    IFS=',' read -ra MODELS <<< "$TEST_MODELS"
    for model in "${MODELS[@]}"; do
        echo "Testing model: $model"
        python -c "
from utils import test_model
test_model('$model', num_samples=$NUM_SAMPLES)
"
        echo "Completed: $model"
        echo "---"
    done
fi

echo "All tests completed!"