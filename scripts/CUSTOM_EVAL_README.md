# Custom Gemini Evaluation Pipeline

This is a simplified, custom evaluation pipeline that bypasses the complexity of the AlpacaEval framework. It directly calls the Gemini API to evaluate model outputs and computes win rates.

## Overview

The pipeline:
1. **Loads model outputs** from a JSON file (e.g., `results_Qwen_Qwen3-8B-Base.json`)
2. **Automatically finds GPT-4 reference annotations** in the results directory
3. **Formats prompts** for Gemini using the same template as AlpacaEval
4. **Calls Gemini API** to rate outputs (A vs B)
5. **Parses responses** to extract preferences (1.0 for A, 2.0 for B)
6. **Computes and displays win rate** automatically
7. **Optionally saves results** in AlpacaEval format

## Requirements

```bash
pip install google-genai python-dotenv tqdm
```

Make sure your `.env` file has:
```
GOOGLE_API_KEY='your-api-key-here'
```

## Usage

### Simple Usage (Just provide model outputs)

```bash
python scripts/custom_gemini_eval.py alpaca_eval_results/results_Qwen_Qwen3-8B-Base.json
```

That's it! The script will:
- Automatically find GPT-4 annotations in `alpaca_eval_results/alpaca_eval_gpt4_turbo_fn/annotations.json`
- Evaluate all examples
- Display win rate statistics
- Optionally save results to `alpaca_eval_results/results_Qwen_Qwen3-8B-Base_gemini/annotations.json`

### Arguments

1. **Model outputs JSON** (required): The file containing model outputs to evaluate
2. **Output path** (optional): Where to save the results. If not specified, auto-generates: `{model_name}_gemini/annotations.json`

### Example

```bash
# Evaluate and see win rate
python scripts/custom_gemini_eval.py alpaca_eval_results/results_Qwen_Qwen3-8B-Base.json

# Output:
# Loading model outputs from ...
# Loading reference outputs from ...
# Evaluating 805 examples...
# 
# ============================================================
# Win Rate Statistics
# ============================================================
# Win Rate:        0.4523 (45.23%)
# Standard Error:  0.0176
# Total Valid:     805
# Wins:            364
# Losses:          441
# Ties:            0
# Failed Parsing:  0
# ============================================================
```

## Output Format

The output JSON follows the same format as AlpacaEval annotations:

```json
[
  {
    "instruction": "...",
    "output_1": "...",  // Reference output (GPT-4)
    "generator_1": "gpt4_1106_preview",
    "output_2": "...",  // Model output being evaluated
    "generator_2": "Qwen/Qwen3-8B-Base",
    "annotator": "gemini_judge",
    "preference": 1.0,  // 1.0 = reference wins, 2.0 = model wins, 1.5 = tie
    "preference_raw_completion": "A",  // Raw Gemini response
    "preference_date": "2026-01-24T..."
  }
]
```

## Advantages

1. **Simpler**: No complex framework dependencies
2. **Direct control**: Easy to modify prompts, parsing logic, etc.
3. **Debugging**: Raw completions are saved for inspection
4. **Fast iteration**: Quick to test changes

## Troubleshooting

### Parsing Failures

If you see many `preference: null` values, check the `preference_raw_completion` field to see what Gemini actually returned. You can then adjust the `parse_preference()` function in `custom_gemini_eval.py`.

### Missing Reference Outputs

If you see warnings about missing reference outputs, make sure the instruction text matches exactly between the model outputs and reference annotations.

### API Errors

Make sure your `GOOGLE_API_KEY` is set correctly in `.env` and that you have API quota available.

## Example Workflow

```bash
# Just run the evaluation - win rate is computed automatically!
python scripts/custom_gemini_eval.py alpaca_eval_results/results_Qwen_Qwen3-8B-Base.json

# If you want to inspect failed parsing later (if any)
python3 -c "
import json
with open('alpaca_eval_results/results_Qwen_Qwen3-8B-Base_gemini/annotations.json') as f:
    data = json.load(f)
failed = [x for x in data if x['preference'] is None]
print(f'Failed: {len(failed)}/{len(data)}')
for i, x in enumerate(failed[:5]):
    print(f\"\\n{i+1}. {x['preference_raw_completion'][:100]}\")
"
```
