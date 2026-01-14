# DPO Training Pipeline

## Core Pipeline Files

### 1. Data Generation
- **`generate_dpo_data.py`**: Generates DPO training data from alpaca dataset using a model
  - Automatically called by training script if model-specific data doesn't exist
  - Outputs: `datasets/dpo/model_generated/{model_name}.jsonl`

### 2. Training
- **`dpo_training.py`**: Main training script
  - Automatically generates data if needed
  - Trains model with DPO loss
  - Generates likelihood plots
  - Tests model and computes ASR
  - Outputs: Model checkpoints, likelihood plots, test results

### 3. Supporting Modules
- **`loss.py`**: Custom loss functions (DPO, IPO, TDPO, BDPO, SimPO, RePO) and CustomDPOTrainer
- **`custom_dataset.py`**: DPODataset class for loading and processing DPO data
- **`config.py`**: Configuration constants (attack sentences, model configs, etc.)
- **`utils.py`**: Utility functions (model loading, text generation, testing)
- **`setup.py`**: Environment setup for Colab/local environments

### 4. Scripts
- **`scripts/run_dpo_training.sh`**: Bash script to run DPO training
- **`scripts/run_test.sh`**: Bash script to test models

## Pipeline Flow

1. **Data Generation** (automatic if needed):
   ```bash
   python generate_dpo_data.py --model_path "Qwen/Qwen3-0.6B"
   ```
   - Loads alpaca dataset
   - Generates responses using the model
   - Creates DPO format data with injection sentences
   - Saves to `datasets/dpo/model_generated/{model_name}.jsonl`

2. **Training**:
   ```bash
   python dpo_training.py --model_path "Qwen/Qwen3-0.6B" --loss_type "dpo"
   ```
   - Checks for model-specific data (generates if missing)
   - Loads model and tokenizer
   - Trains with selected loss function
   - Records likelihoods at logging steps
   - Generates three plots: chosen likelihood, rejected likelihood, margin
   - Tests model and computes ASR

3. **Results**:
   - Model checkpoints: `model_outputs/{output_dir}/`
   - Likelihood plots: `model_outputs/{output_dir}/likelihoods_plot.png`
   - Test results: `test/{model_path}/test_output.json`

## Other Files (Not Part of Core DPO Pipeline)

These files serve different purposes or are separate projects:
- `struq.py` - StruQ-style structured instruction tuning (SFT) to defend against prompt injection
- `eval.py` - AlpacaEval evaluation (different from DPO testing)
- `generate_alpaca_eval_outputs.py` - AlpacaEval evaluation outputs
- `data.py` - SST-2 dataset preparation (not used in DPO)
- `DOOR-Alignment/` - Separate project directory
- `models/qwen.py` - Used by utils for Qwen model wrapper
- `models/llama.py`, `models/mistral.py`, `models/phi3.py`, `models/gpt.py` - Other model wrappers (may be unused)

## Deleted Files (Consolidated)

These files were removed as duplicates/old versions:
- ~~`generate_data.py`~~ - Old data processing (logic moved to utils/custom_dataset)
- ~~`test.py`~~ - Old test implementation (replaced by `utils.test_model`)
- ~~`test_model.py`~~ - Old test implementation (replaced by `utils.test_model`)

## Notes

- The pipeline automatically handles data generation, so manual data generation is optional
- Model-specific data is cached in `datasets/dpo/model_generated/` to avoid regeneration
- All loss functions track likelihoods for visualization
- Testing is integrated into the training script

