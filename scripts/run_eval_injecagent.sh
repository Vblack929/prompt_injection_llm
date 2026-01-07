#!/bin/bash
# Run InjecAgent evaluation
# Usage: ./run_eval_injecagent.sh [MODEL_PATH] [SETTING] [PROMPT_TYPE] [NUM_SAMPLES] [ONLY_FIRST_STEP]

MODEL_PATH="${1:-Qwen/Qwen3-0.6B}"
SETTING="${2:-base}"
PROMPT_TYPE="${3:-InjecAgent}"
NUM_SAMPLES="${4:-}"
ONLY_FIRST_STEP="${5:-false}"

echo "Running InjecAgent evaluation..."
echo "Model: $MODEL_PATH"
echo "Setting: $SETTING"
echo "Prompt type: $PROMPT_TYPE"
echo "Samples: ${NUM_SAMPLES:-all}"
echo "Only first step: $ONLY_FIRST_STEP"
echo ""

if [ "$ONLY_FIRST_STEP" = "true" ]; then
    python -m eval.injecagent_cli \
        --model_path "$MODEL_PATH" \
        --setting "$SETTING" \
        --prompt_type "$PROMPT_TYPE" \
        --num_samples "$NUM_SAMPLES" \
        --only_first_step
else
    python -m eval.injecagent_cli \
        --model_path "$MODEL_PATH" \
        --setting "$SETTING" \
        --prompt_type "$PROMPT_TYPE" \
        --num_samples "$NUM_SAMPLES"
fi

