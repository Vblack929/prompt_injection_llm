# Batch Evaluation Script

Run evaluation on all JSON files in `alpaca_eval_results/eval_results` and collect win rates.

## Usage

```bash
# Run batch evaluation (defaults to alpaca_eval_results/eval_results)
python scripts/batch_eval.py

# Or specify custom directory
python scripts/batch_eval.py /path/to/eval_results [output_summary.txt]
```

## Output

The script will:
1. Process all JSON files in the eval_results directory
2. Run Gemini evaluation on each file
3. Collect win rate statistics
4. Write a summary to `alpaca_eval_results/gemini_eval_summary.txt`

## Summary File Format

The summary file contains:
- A table with file names and win rates (sorted by win rate)
- Detailed statistics for each file
- Error messages for any failed evaluations

Example output:
```
================================================================================
GEMINI EVALUATION RESULTS SUMMARY
================================================================================
Generated: 2026-01-24T12:00:00
Total files: 14

File Name                                                  Win Rate    Std Err    Wins    Losses  Total   
--------------------------------------------------------------------------------
results_Qwen_Qwen3-8B-Base.json                           0.4523      0.0176     364     441     805    
results_Llama-3.1-8B-Instruct_dpo_20260119_111253.json   0.4231      0.0189     340     463     803    
...
```

## Notes

- The script uses the GPT-4 annotations from `alpaca_eval_results/alpaca_eval_gpt4_turbo_fn/annotations.json`
- Individual evaluation results are NOT saved (only the summary)
- Progress is shown for each file being processed
