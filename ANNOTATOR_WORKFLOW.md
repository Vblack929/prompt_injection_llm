# AlpacaEval Annotator Workflow and Debugging Guide

## Annotator Workflow

The annotator follows this workflow:

1. **Preprocessing** (`_preprocess`)
   - Applies processors (e.g., randomization, padding)
   - Shuffles examples if `is_shuffle=True`
   - Prepares dataframe for annotation

2. **Prompt Generation** (`_make_prompts`)
   - Formats prompts using the prompt template
   - Replaces placeholders like `{instruction}`, `{output_1}`, `{output_2}`
   - Returns list of formatted prompts

3. **Completion Generation** (`fn_completions`)
   - Calls the completion function (e.g., `openai_completions`, `google_completions`)
   - Sends prompts to the API (OpenAI, Google, etc.)
   - Returns raw completions with metadata (price, time, etc.)

4. **Metadata Addition** (`_add_metadata_to_completions_`)
   - Adds timestamp (`date`)
   - Adds version information
   - Stores in completion dict

5. **Completion Storage**
   - Raw completions stored in `completion_column` (default: `"raw_completion"`)
   - All completion metadata (price, time) stored with prefix
   - Available for debugging and re-parsing

6. **Parsing** (`_parse_completions`)
   - Parses raw completions using `fn_completion_parser`
   - Converts raw text to structured annotations (e.g., preference scores)
   - Returns parsed annotations and cleaned completions

7. **Postprocessing** (`_postprocess`)
   - Applies postprocessors
   - Finalizes dataframe

8. **Saving**
   - Annotations saved to `annotations.json` with all columns
   - Includes: instruction, outputs, preference, raw_completion, metadata

## How to Enable Raw Completion Storage for Debugging

Raw completions are **automatically stored** if `completion_column` is not `None` (default: `"raw_completion"`).

### What Gets Stored

The `annotations.json` file includes:
- `instruction`: The instruction/prompt
- `output_1`, `output_2`: The model outputs being compared
- `generator_1`, `generator_2`: Model names
- `preference`: Parsed preference (1, 2, or null if parsing failed)
- `raw_completion`: **Raw text response from the annotator** (if `completion_column` is set)
- `{annotator}_date`: Timestamp
- `{annotator}_version`: Version info
- `{annotator}_price_per_example`: Cost per example
- `{annotator}_time_per_example`: Time per example

### Current Issue

Looking at your `annotations.json`, the `preference` field is `null`, which means:
1. The raw completion was generated (Gemini responded)
2. But parsing failed (couldn't extract "A" or "B")

### Debugging Steps

1. **Check Raw Completions**: Look for `raw_completion` or `{annotator}_raw_completion` field in annotations.json
2. **Check Parsing Errors**: Look for error messages in logs about parsing failures
3. **Verify Parser Configuration**: Ensure `completion_parser_kwargs` has `outputs_to_match` for regex_parser

### Example: Inspecting Raw Completions

```python
import json
import pandas as pd

# Load annotations
with open('alpaca_eval_results/results_Qwen_Qwen3-1.7B_gemini-2.5-flash/gemini-2.5-flash/annotations.json') as f:
    annotations = json.load(f)

# Check first few raw completions
for i, ann in enumerate(annotations[:5]):
    print(f"\n=== Example {i+1} ===")
    print(f"Instruction: {ann.get('instruction', 'N/A')[:100]}...")
    print(f"Preference: {ann.get('preference', 'NULL')}")
    # Check for raw completion field (might be named differently)
    for key in ann.keys():
        if 'raw' in key.lower() or 'completion' in key.lower():
            print(f"{key}: {ann[key][:200]}...")  # First 200 chars
```

### Current Status

**Raw completions should be automatically saved** in the `annotations.json` file under the field:
- `{annotation_key}_raw_completion` (e.g., `preference_raw_completion` for pairwise evaluation)

However, if they're missing, it means:
1. The `completion_column` wasn't properly set or matched
2. The column was filtered out during saving
3. There was an error during annotation that prevented saving

### Debugging: Check What Gemini Actually Returned

To see what Gemini returned (even if parsing failed), you can:

1. **Check the logs** - Look for error messages showing the raw completion
2. **Re-run with logging** - The code logs parsing errors with the raw completion content
3. **Add debug output** - Modify the code to print raw completions before parsing

### Quick Debug Script

```python
import json

# Load annotations
with open('alpaca_eval_results/results_Qwen_Qwen3-1.7B_gemini-2.5-flash/gemini-2.5-flash/annotations.json') as f:
    annotations = json.load(f)

# Find examples with null preference (parsing failed)
failed = [a for a in annotations if a.get('preference') is None]
print(f"Found {len(failed)} examples with failed parsing out of {len(annotations)} total")

# Check if raw completions exist
has_raw = [a for a in annotations if any('raw' in k.lower() or 'completion' in k.lower() 
                                        for k in a.keys() if k != 'instruction')]
print(f"Found {len(has_raw)} examples with raw completion fields")

# Show all keys in first annotation
if annotations:
    print(f"\nAll keys in annotations: {list(annotations[0].keys())}")
```
