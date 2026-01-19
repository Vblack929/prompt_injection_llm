#!/bin/bash
#
# Run SEP evaluation (instruction-data separation) across:
# - no defense
# - prompt-defense baselines (sandwich / instructional / reminder)
#
# Models are discovered from this repo's training outputs as PEFT adapter directories
# under model_outputs/ (directories containing adapter_config.json), excluding checkpoints.
#
# Usage:
#   bash scripts/run_sep_eval_all_methods_trained_models.sh
#
# Optional env vars:
#   SEP_NUM_SAMPLES=208
#   SEP_DATASET_PATH=datasets/sep_dataset/sep_probe_examples.json
#   SEP_OUT_ROOT=model_outputs/sep_eval
#

set -euo pipefail

SEP_NUM_SAMPLES="${SEP_NUM_SAMPLES:-208}"
SEP_DATASET_PATH="${SEP_DATASET_PATH:-datasets/sep_dataset/sep_probe_examples.json}"
SEP_OUT_ROOT="${SEP_OUT_ROOT:-model_outputs/sep_eval}"

DEFENSES=("none" "sandwich" "instructional" "reminder")

echo "=========================================="
echo "SEP eval (all methods) on trained adapters"
echo "Dataset: $SEP_DATASET_PATH"
echo "Num samples: $SEP_NUM_SAMPLES"
echo "Out root: $SEP_OUT_ROOT"
echo "Defenses: ${DEFENSES[*]}"
echo "=========================================="

# Find adapter dirs (exclude checkpoints)
mapfile -t ADAPTER_CONFIGS < <(find model_outputs -maxdepth 3 -name adapter_config.json -not -path \"*checkpoint-*/*\" | sort)

if [ "${#ADAPTER_CONFIGS[@]}" -eq 0 ]; then
  echo "No adapter_config.json found under model_outputs/. Nothing to run."
  exit 1
fi

TOTAL=${#ADAPTER_CONFIGS[@]}
IDX=0

for CFG in "${ADAPTER_CONFIGS[@]}"; do
  IDX=$((IDX + 1))
  MODEL_DIR="$(dirname "$CFG")"
  TAG="$(echo "$MODEL_DIR" | sed 's#^model_outputs/##' | sed 's#/#__#g')"

  # Heuristic batch sizing from path name
  BATCH_SIZE=8
  case "$MODEL_DIR" in
    *"1.7B"*) BATCH_SIZE=4 ;;
    *"3B"*) BATCH_SIZE=2 ;;
    *"8B"*|*"70B"*|*"34B"*) BATCH_SIZE=1 ;;
  esac

  echo ""
  echo "=========================================="
  echo "[$IDX/$TOTAL] $MODEL_DIR"
  echo "  batch_size=$BATCH_SIZE"
  echo "=========================================="

  for DEF in "${DEFENSES[@]}"; do
    OUT_DIR="${SEP_OUT_ROOT}/${TAG}/${DEF}"
    mkdir -p "$OUT_DIR"
    OUT_PATH="${OUT_DIR}/sep_results.json"

    echo "[SEP] defense=$DEF out=$OUT_PATH"
    python -m eval.sep_cli \
      --model_path "$MODEL_DIR" \
      --dataset_path "$SEP_DATASET_PATH" \
      --num_samples "$SEP_NUM_SAMPLES" \
      --batch_size "$BATCH_SIZE" \
      --defense "$DEF" \
      --output_path "$OUT_PATH" \
      > "${OUT_DIR}/stdout.json"
  done
done

echo ""
echo "=========================================="
echo "Done. Results under: $SEP_OUT_ROOT"
echo "=========================================="

