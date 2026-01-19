#!/bin/bash

# Run all evaluations (ASR, AlpacaEval, InjecAgent) for one model
# Usage: ./run_all_evals_one_model.sh [MODEL_PATH] [NUM_SAMPLES] [OPTIONS]
#
# Options:
#   --skip-asr          Skip ASR evaluation
#   --skip-alpaca       Skip AlpacaEval evaluation
#   --skip-injecagent   Skip InjecAgent evaluation
#   --alpaca-generate-only  Only generate outputs for AlpacaEval (skip scoring)
#   --injecagent-setting SETTING  InjecAgent setting: base (default) or enhanced
#   --injecagent-prompt PROMPT    InjecAgent prompt type: InjecAgent (default) or hwchase17_react

MODEL_PATH="${1:-Qwen/Qwen3-0.6B}"
NUM_SAMPLES="${2:-}"

# Parse optional flags
SKIP_ASR=false
SKIP_ALPACA=false
SKIP_INJECAGENT=false
ALPACA_GENERATE_ONLY=false
INJECAGENT_SETTING="base"
INJECAGENT_PROMPT="InjecAgent"

shift 2 2>/dev/null || shift 1 2>/dev/null || true

while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-asr)
            SKIP_ASR=true
            shift
            ;;
        --skip-alpaca)
            SKIP_ALPACA=true
            shift
            ;;
        --skip-injecagent)
            SKIP_INJECAGENT=true
            shift
            ;;
        --alpaca-generate-only)
            ALPACA_GENERATE_ONLY=true
            shift
            ;;
        --injecagent-setting)
            INJECAGENT_SETTING="$2"
            shift 2
            ;;
        --injecagent-prompt)
            INJECAGENT_PROMPT="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [MODEL_PATH] [NUM_SAMPLES] [OPTIONS]"
            exit 1
            ;;
    esac
done

echo "=========================================="
echo "Running All Evaluations for Model"
echo "=========================================="
echo "Model: $MODEL_PATH"
echo "Samples: ${NUM_SAMPLES:-all}"
echo ""
echo "Evaluations to run:"
echo "  - ASR: $([ "$SKIP_ASR" = true ] && echo "SKIPPED" || echo "ENABLED")"
echo "  - AlpacaEval: $([ "$SKIP_ALPACA" = true ] && echo "SKIPPED" || echo "ENABLED")"
echo "  - InjecAgent: $([ "$SKIP_INJECAGENT" = true ] && echo "SKIPPED" || echo "ENABLED")"
echo "=========================================="
echo ""

# Track results
RESULTS_DIR="evaluation_results"
mkdir -p "$RESULTS_DIR"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
MODEL_NAME=$(basename "$MODEL_PATH" | tr '/' '_')
SUMMARY_FILE="$RESULTS_DIR/${MODEL_NAME}_${TIMESTAMP}_summary.txt"

echo "Results will be saved to: $RESULTS_DIR"
echo "Summary file: $SUMMARY_FILE"
echo ""

# Function to run ASR evaluation
run_asr() {
    echo ""
    echo "=========================================="
    echo "1. Running ASR Evaluation"
    echo "=========================================="
    echo ""
    
    if [ -n "$NUM_SAMPLES" ]; then
        python -m eval.asr_cli \
            --model_path "$MODEL_PATH" \
            --num_samples "$NUM_SAMPLES" \
            --batch_size 8
    else
        python -m eval.asr_cli \
            --model_path "$MODEL_PATH" \
            --batch_size 8
    fi
    
    if [ $? -eq 0 ]; then
        echo "✓ ASR evaluation completed"
        return 0
    else
        echo "✗ ASR evaluation failed"
        return 1
    fi
}

# Function to run AlpacaEval evaluation
run_alpaca() {
    echo ""
    echo "=========================================="
    echo "2. Running AlpacaEval Evaluation"
    echo "=========================================="
    echo ""
    
    if [ "$ALPACA_GENERATE_ONLY" = true ]; then
        echo "Mode: Generate outputs only (no scoring)"
        if [ -n "$NUM_SAMPLES" ]; then
            python -m eval.alpaca_eval_cli \
                --model_path "$MODEL_PATH" \
                --num_samples "$NUM_SAMPLES" \
                --batch_size 8 \
                --generate_only
        else
            python -m eval.alpaca_eval_cli \
                --model_path "$MODEL_PATH" \
                --batch_size 8 \
                --generate_only
        fi
    else
        echo "Mode: Full evaluation (generate + score)"
        if [ -n "$NUM_SAMPLES" ]; then
            python -m eval.alpaca_eval_cli \
                --model_path "$MODEL_PATH" \
                --num_samples "$NUM_SAMPLES" \
                --batch_size 8
        else
            python -m eval.alpaca_eval_cli \
                --model_path "$MODEL_PATH" \
                --batch_size 8
        fi
    fi
    
    if [ $? -eq 0 ]; then
        echo "✓ AlpacaEval evaluation completed"
        return 0
    else
        echo "✗ AlpacaEval evaluation failed"
        return 1
    fi
}

# Function to run InjecAgent evaluation
run_injecagent() {
    echo ""
    echo "=========================================="
    echo "3. Running InjecAgent Evaluation"
    echo "=========================================="
    echo "Setting: $INJECAGENT_SETTING"
    echo "Prompt Type: $INJECAGENT_PROMPT"
    echo ""
    
    if [ -n "$NUM_SAMPLES" ]; then
        python -m eval.injecagent_cli \
            --model_path "$MODEL_PATH" \
            --setting "$INJECAGENT_SETTING" \
            --prompt_type "$INJECAGENT_PROMPT" \
            --num_samples "$NUM_SAMPLES" \
            --batch_size 8
    else
        python -m eval.injecagent_cli \
            --model_path "$MODEL_PATH" \
            --setting "$INJECAGENT_SETTING" \
            --prompt_type "$INJECAGENT_PROMPT" \
            --batch_size 8
    fi
    
    if [ $? -eq 0 ]; then
        echo "✓ InjecAgent evaluation completed"
        return 0
    else
        echo "✗ InjecAgent evaluation failed"
        return 1
    fi
}

# Run evaluations
ASR_RESULT=""
ALPACA_RESULT=""
INJECAGENT_RESULT=""

if [ "$SKIP_ASR" = false ]; then
    if run_asr; then
        ASR_RESULT="SUCCESS"
    else
        ASR_RESULT="FAILED"
    fi
else
    ASR_RESULT="SKIPPED"
fi

if [ "$SKIP_ALPACA" = false ]; then
    if run_alpaca; then
        ALPACA_RESULT="SUCCESS"
    else
        ALPACA_RESULT="FAILED"
    fi
else
    ALPACA_RESULT="SKIPPED"
fi

if [ "$SKIP_INJECAGENT" = false ]; then
    if run_injecagent; then
        INJECAGENT_RESULT="SUCCESS"
    else
        INJECAGENT_RESULT="FAILED"
    fi
else
    INJECAGENT_RESULT="SKIPPED"
fi

# Generate summary
echo ""
echo "=========================================="
echo "Evaluation Summary"
echo "=========================================="
echo "Model: $MODEL_PATH"
echo "Timestamp: $TIMESTAMP"
echo ""
echo "Results:"
echo "  ASR:        $ASR_RESULT"
echo "  AlpacaEval: $ALPACA_RESULT"
echo "  InjecAgent: $INJECAGENT_RESULT"
echo "=========================================="

# Save summary to file
cat > "$SUMMARY_FILE" << EOF
Evaluation Summary
==================
Model: $MODEL_PATH
Timestamp: $TIMESTAMP
Samples: ${NUM_SAMPLES:-all}

Results:
  ASR:        $ASR_RESULT
  AlpacaEval: $ALPACA_RESULT
  InjecAgent: $INJECAGENT_RESULT

Options:
  AlpacaEval Generate Only: $ALPACA_GENERATE_ONLY
  InjecAgent Setting: $INJECAGENT_SETTING
  InjecAgent Prompt: $INJECAGENT_PROMPT
EOF

echo ""
echo "Summary saved to: $SUMMARY_FILE"
echo ""

