#!/bin/bash
#
# Run alpaca_eval evaluation on selected JSON results files
#
# Usage:
#   # List available files
#   bash scripts/run_alpaca_eval.sh --list
#
#   # Process specific files
#   bash scripts/run_alpaca_eval.sh --files file1.json file2.json
#
#   # Process all files
#   bash scripts/run_alpaca_eval.sh --all
#
#   # Process files with custom config
#   bash scripts/run_alpaca_eval.sh --files file1.json --annotators_config 'weighted_alpaca_eval_gpt4_turbo'
#
# What the command does:
#   Uses the local alpaca_eval repo to run evaluations. When you run:
#   python -m alpaca_eval.main --model_outputs <path> --annotators_config <config> --output_path <dir>
#   It creates a directory with:
#     - leaderboard.csv: Contains win_rate, standard_error, avg_length, n_wins, n_total, etc.
#     - annotations.json: Contains detailed preference annotations for each example
#
# Arguments:
#   --files <file1> [file2] ...: Specific JSON files to process (relative to results_dir or absolute paths)
#   --all: Process all JSON files in results directory (default if no files specified)
#   --list: List available JSON files and exit
#   --annotators_config: Annotators config name (default: alpaca_eval_gpt4_turbo_fn)
#   --results_dir: Directory containing JSON result files (default: alpaca_eval_results)
#   --output_base: Base directory for output (default: alpaca_eval_results)
#
# Environment:
#   OPENAI_API_KEY: Required for OpenAI-based annotators (GPT-4, etc.)
#   GOOGLE_API_KEY: Required for Google-based annotators (Gemini models)

set -euo pipefail

# Default values
ANNOTATORS_CONFIG="${ANNOTATORS_CONFIG:-gemini-2.5-flash}"
RESULTS_DIR="${RESULTS_DIR:-alpaca_eval_results}"
OUTPUT_BASE="${OUTPUT_BASE:-alpaca_eval_results}"
PROCESS_ALL=false
LIST_ONLY=false
SELECTED_FILES=()

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --annotators_config)
            ANNOTATORS_CONFIG="$2"
            shift 2
            ;;
        --results_dir)
            RESULTS_DIR="$2"
            shift 2
            ;;
        --output_base)
            OUTPUT_BASE="$2"
            shift 2
            ;;
        --files)
            shift
            # Collect all files until next option or end
            while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do
                SELECTED_FILES+=("$1")
                shift
            done
            ;;
        --all)
            PROCESS_ALL=true
            shift
            ;;
        --list)
            LIST_ONLY=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: bash scripts/run_alpaca_eval.sh [--files <file1> [file2] ...] [--all] [--list] [--annotators_config <config>]"
            exit 1
            ;;
    esac
done

# Check if results directory exists
if [ ! -d "$RESULTS_DIR" ]; then
    echo "Error: Results directory not found: $RESULTS_DIR"
    exit 1
fi

# Find all JSON files in results directory (excluding subdirectories)
ALL_JSON_FILES=($(find "$RESULTS_DIR" -maxdepth 1 -name "*.json" -type f | sort))

if [ ${#ALL_JSON_FILES[@]} -eq 0 ]; then
    echo "Error: No JSON files found in $RESULTS_DIR"
    exit 1
fi

# Handle --list option
if [ "$LIST_ONLY" = true ]; then
    echo "Available JSON files in $RESULTS_DIR:"
    echo "=========================================="
    for i in "${!ALL_JSON_FILES[@]}"; do
        echo "$((i+1)). $(basename "${ALL_JSON_FILES[$i]}")"
    done
    echo "=========================================="
    echo ""
    echo "To process specific files, use:"
    echo "  bash scripts/run_alpaca_eval.sh --files file1.json file2.json"
    echo ""
    echo "To process all files, use:"
    echo "  bash scripts/run_alpaca_eval.sh --all"
    exit 0
fi

# Determine which files to process
if [ ${#SELECTED_FILES[@]} -gt 0 ]; then
    # Process selected files
    JSON_FILES=()
    for FILE in "${SELECTED_FILES[@]}"; do
        # Check if it's an absolute path
        if [[ "$FILE" == /* ]]; then
            if [ -f "$FILE" ]; then
                JSON_FILES+=("$FILE")
            else
                echo "Warning: File not found: $FILE"
            fi
        else
            # Try relative to results_dir
            FULL_PATH="$RESULTS_DIR/$FILE"
            if [ -f "$FULL_PATH" ]; then
                JSON_FILES+=("$FULL_PATH")
            elif [ -f "$FILE" ]; then
                # Try current directory
                JSON_FILES+=("$FILE")
            else
                echo "Warning: File not found: $FILE (checked $FULL_PATH and $FILE)"
            fi
        fi
    done
    
    if [ ${#JSON_FILES[@]} -eq 0 ]; then
        echo "Error: No valid files found to process"
        exit 1
    fi
elif [ "$PROCESS_ALL" = true ] || [ ${#SELECTED_FILES[@]} -eq 0 ]; then
    # Process all files (default behavior)
    JSON_FILES=("${ALL_JSON_FILES[@]}")
fi

# Check for API keys (OpenAI for GPT models, Google for Gemini models)
if [[ "$ANNOTATORS_CONFIG" == *"gemini"* ]] || [[ "$ANNOTATORS_CONFIG" == *"Gemini"* ]]; then
    if [ -z "${GOOGLE_API_KEY:-}" ]; then
        echo "Error: GOOGLE_API_KEY is not set in environment (required for Gemini annotators)"
        exit 1
    fi
else
    if [ -z "${OPENAI_API_KEY:-}" ]; then
        echo "Error: OPENAI_API_KEY is not set in environment (required for OpenAI annotators)"
        exit 1
    fi
fi

# Set up path to local alpaca_eval repo
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
ALPACA_EVAL_REPO="$PROJECT_ROOT/alpaca_eval/src"

# Check if local repo exists
if [ ! -d "$ALPACA_EVAL_REPO" ]; then
    echo "Error: Local alpaca_eval repo not found at: $ALPACA_EVAL_REPO"
    echo "Please ensure the alpaca_eval directory exists in the project root"
    exit 1
fi

# Export PYTHONPATH to use local repo
export PYTHONPATH="$ALPACA_EVAL_REPO${PYTHONPATH:+:$PYTHONPATH}"

echo "=========================================="
echo "Running AlpacaEval Evaluation"
echo "=========================================="
echo "Results directory: $RESULTS_DIR"
echo "Annotators config: $ANNOTATORS_CONFIG"
echo "Output base directory: $OUTPUT_BASE"
echo "Files to process: ${#JSON_FILES[@]}"
echo ""
echo "Files:"
for FILE in "${JSON_FILES[@]}"; do
    echo "  - $(basename "$FILE")"
done
echo "=========================================="
echo ""

# Track success/failure
SUCCESS_COUNT=0
FAIL_COUNT=0
FAILED_FILES=()

# Process each JSON file
for JSON_FILE in "${JSON_FILES[@]}"; do
    # Extract base filename without extension
    BASENAME=$(basename "$JSON_FILE" .json)
    
    # Create output directory name based on filename and annotators config
    OUTPUT_DIR="$OUTPUT_BASE/${BASENAME}_${ANNOTATORS_CONFIG}"
    
    echo "----------------------------------------"
    echo "Processing: $(basename "$JSON_FILE")"
    echo "Output directory: $OUTPUT_DIR"
    echo "----------------------------------------"
    
    # Create output directory if it doesn't exist
    mkdir -p "$OUTPUT_DIR"
    
    # Run alpaca_eval using local repo
    # Use python -m to run from the local source with explicit 'evaluate' command
    if python -m alpaca_eval.main evaluate \
        --model_outputs="$JSON_FILE" \
        --annotators_config="$ANNOTATORS_CONFIG" \
        --output_path="$OUTPUT_DIR"; then
        
        SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
        echo ""
        echo "✓ Evaluation completed successfully!"
        
        # Display summary if leaderboard exists
        LEADERBOARD="$OUTPUT_DIR/leaderboard.csv"
        if [ -f "$LEADERBOARD" ]; then
            echo "Leaderboard summary:"
            head -3 "$LEADERBOARD" | column -t -s, | head -2
            echo "Full leaderboard: $LEADERBOARD"
        fi
    else
        FAIL_COUNT=$((FAIL_COUNT + 1))
        FAILED_FILES+=("$JSON_FILE")
        echo ""
        echo "✗ Evaluation failed for: $(basename "$JSON_FILE")"
    fi
    echo ""
done

# Final summary
echo "=========================================="
echo "Evaluation Summary"
echo "=========================================="
echo "Total files processed: ${#JSON_FILES[@]}"
echo "Successful: $SUCCESS_COUNT"
echo "Failed: $FAIL_COUNT"
echo ""

if [ $FAIL_COUNT -gt 0 ]; then
    echo "Failed files:"
    for FILE in "${FAILED_FILES[@]}"; do
        echo "  - $FILE"
    done
    echo ""
fi

echo "Results saved to: $OUTPUT_BASE"
echo "=========================================="

# Exit with error if any evaluations failed
if [ $FAIL_COUNT -gt 0 ]; then
    exit 1
fi
