# Multi-Model Training and Evaluation Scripts

This document describes two new scripts for batch training and comprehensive evaluation.

## 1. Multi-Model DPO Training Script

**File**: `run_dpo_training_multi_model.sh`

### Purpose
Run DPO training with different loss functions for multiple models without evaluation. Useful for training multiple model/loss combinations in batch.

### Usage

```bash
./scripts/run_dpo_training_multi_model.sh
```

### Configuration

Edit the script to customize:

```bash
# Models to train
MODELS=(
    "Qwen/Qwen3-0.6B"
    "Qwen/Qwen3-1.7B"
    "meta-llama/Llama-3.2-1B-Instruct"
)

# Loss functions to use
LOSS_FUNCTIONS=(
    "dpo"
    "simpo"
    "simper"
    "ipo"
    "tdpo"
    "bdpo"
    "repo"
)

# Training hyperparameters
EPOCHS=3
BATCH_SIZE=4
LEARNING_RATE=5e-5
NUM_SAMPLES=500
BETA=0.5
GAMMA=0.5
ALPHA=0.5
LAMBDA_MIX=0.5
```

### Features

- **Batch Training**: Trains all combinations of models × loss functions
- **Progress Tracking**: Shows progress (e.g., [2/6] Training: Qwen/Qwen3-0.6B with simpo)
- **Error Handling**: Continues with next combination if one fails
- **No Evaluation**: Focuses only on training (faster execution)

### Example Output

```
==========================================
Multi-Model DPO Training (No Evaluation)
==========================================
Models: Qwen/Qwen3-0.6B Qwen/Qwen3-1.7B
Loss Functions: dpo simpo simper
Total combinations: 6
==========================================

[1/6] Training: Qwen/Qwen3-0.6B with dpo
...
✓ Successfully trained Qwen/Qwen3-0.6B with dpo

[2/6] Training: Qwen/Qwen3-0.6B with simpo
...
```

### Output Locations

Each training run saves to:
```
model_outputs/{model_name}_{loss_type}_{timestamp}/
```

## 2. All Evaluations for One Model Script

**File**: `run_all_evals_one_model.sh`

### Purpose
Run all three evaluation methods (ASR, AlpacaEval, InjecAgent) for a single model sequentially.

### Usage

```bash
# Basic usage
./scripts/run_all_evals_one_model.sh [MODEL_PATH] [NUM_SAMPLES]

# Examples
./scripts/run_all_evals_one_model.sh "Qwen/Qwen3-0.6B" 100
./scripts/run_all_evals_one_model.sh "model_outputs/Qwen3-0.6B_dpo_20240101_120000" 50
```

### Options

```bash
# Skip specific evaluations
./scripts/run_all_evals_one_model.sh MODEL_PATH NUM_SAMPLES --skip-asr
./scripts/run_all_evals_one_model.sh MODEL_PATH NUM_SAMPLES --skip-alpaca
./scripts/run_all_evals_one_model.sh MODEL_PATH NUM_SAMPLES --skip-injecagent

# AlpacaEval: Generate outputs only (skip scoring)
./scripts/run_all_evals_one_model.sh MODEL_PATH NUM_SAMPLES --alpaca-generate-only

# InjecAgent: Customize settings
./scripts/run_all_evals_one_model.sh MODEL_PATH NUM_SAMPLES \
    --injecagent-setting enhanced \
    --injecagent-prompt hwchase17_react
```

### Features

- **Comprehensive Evaluation**: Runs all three evaluation methods
- **Flexible Options**: Skip evaluations or customize settings
- **Progress Tracking**: Shows progress for each evaluation
- **Summary Report**: Generates a summary file with results

### Example Output

```
==========================================
Running All Evaluations for Model
==========================================
Model: Qwen/Qwen3-0.6B
Samples: 100

Evaluations to run:
  - ASR: ENABLED
  - AlpacaEval: ENABLED
  - InjecAgent: ENABLED
==========================================

==========================================
1. Running ASR Evaluation
==========================================
...
✓ ASR evaluation completed

==========================================
2. Running AlpacaEval Evaluation
==========================================
...
✓ AlpacaEval evaluation completed

==========================================
3. Running InjecAgent Evaluation
==========================================
...
✓ InjecAgent evaluation completed

==========================================
Evaluation Summary
==========================================
Model: Qwen/Qwen3-0.6B
Timestamp: 20240101_120000

Results:
  ASR:        SUCCESS
  AlpacaEval: SUCCESS
  InjecAgent: SUCCESS
==========================================
```

### Output Locations

- **ASR Results**: `alpaca_eval_results/{model_name}_asr.json`
- **AlpacaEval Results**: `alpaca_eval_results/{model_name}_alpaca_eval/`
- **InjecAgent Results**: `outputs/injecagent/{model_name}_InjecAgent/`
- **Summary File**: `evaluation_results/{model_name}_{timestamp}_summary.txt`

## Workflow Examples

### Example 1: Train Multiple Models, Then Evaluate One

```bash
# Step 1: Train multiple model/loss combinations
./scripts/run_dpo_training_multi_model.sh

# Step 2: Evaluate a specific trained model
./scripts/run_all_evals_one_model.sh \
    "model_outputs/Qwen3-0.6B_dpo_20240101_120000" \
    100
```

### Example 2: Quick Evaluation (Generate Only)

```bash
# Run evaluations but skip scoring for AlpacaEval (faster)
./scripts/run_all_evals_one_model.sh \
    "Qwen/Qwen3-0.6B" \
    50 \
    --alpaca-generate-only
```

### Example 3: Custom Evaluation Suite

```bash
# Run only ASR and InjecAgent (skip AlpacaEval)
./scripts/run_all_evals_one_model.sh \
    "Qwen/Qwen3-0.6B" \
    100 \
    --skip-alpaca \
    --injecagent-setting enhanced
```

## Tips

1. **Training**: Use `run_dpo_training_multi_model.sh` for batch training when you want to train multiple combinations without evaluation overhead.

2. **Evaluation**: Use `run_all_evals_one_model.sh` after training to comprehensively evaluate a model.

3. **Resource Management**: 
   - Training multiple models can be resource-intensive. Consider running fewer combinations or using smaller models.
   - Evaluation can take time, especially AlpacaEval scoring. Use `--alpaca-generate-only` for faster runs.

4. **Model Paths**: 
   - Can use HuggingFace model IDs: `"Qwen/Qwen3-0.6B"`
   - Can use local paths: `"model_outputs/Qwen3-0.6B_dpo_20240101_120000"`
   - Can use adapter directories: `"adapters/my_adapter"`

5. **Parallel Execution**: These scripts run sequentially. For parallel execution, you can modify the scripts or run multiple instances with different configurations.

## Troubleshooting

### Training Fails
- Check GPU memory availability
- Reduce batch size or number of samples
- Verify model paths are correct
- Check HuggingFace token for Llama models

### Evaluation Fails
- Ensure model path is correct and accessible
- Check disk space for output files
- Verify evaluation dependencies are installed
- For InjecAgent, ensure `third_party/InjecAgent` is set up correctly

### Script Permissions
If scripts are not executable:
```bash
chmod +x scripts/run_dpo_training_multi_model.sh
chmod +x scripts/run_all_evals_one_model.sh
```

