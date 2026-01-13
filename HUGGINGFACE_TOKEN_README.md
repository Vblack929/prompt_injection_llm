# HuggingFace Token Configuration

This document explains how HuggingFace tokens are used for accessing Llama models.

## Overview

Llama models on HuggingFace require authentication via a HuggingFace token. This codebase automatically uses a token when loading Llama models.

## Token Configuration

### Environment Variable Required
The HuggingFace token **must** be set via the `HUGGINGFACE_TOKEN` environment variable:

```bash
export HUGGINGFACE_TOKEN="your_token_here"
```

**No default token is provided** - you must set this environment variable before running scripts that use Llama models.

## Automatic Detection

The codebase automatically detects Llama models and applies the token:

1. **Model Detection**: Uses `detect_model_family()` from `model_configs.registry` to identify Llama models
2. **Token Application**: When a Llama model is detected, the token is automatically passed to:
   - `AutoTokenizer.from_pretrained()`
   - `AutoModelForCausalLM.from_pretrained()`

## Files Updated

The following files have been updated to support Llama model token authentication:

1. **`utils.py`**:
   - `load_model_auto()` - Main model loading function
   - Automatically applies token for Llama models

2. **`dpo_training.py`**:
   - `create_ref_model()` - Reference model creation
   - `setup_model_and_tokenizer()` - Model and tokenizer setup

3. **`generate_alpaca_eval_outputs.py`**:
   - `AlpacaEvalGenerator._load_model()` - Model loading for evaluation

## Usage

### Normal Usage (Automatic)
The token is applied automatically - no action needed:

```python
from utils import load_model_auto

# Token is automatically applied for Llama models
model, tokenizer = load_model_auto("meta-llama/Llama-3.2-1B-Instruct")
```

### Using Environment Variable
Set the token before running your scripts:

```bash
export HUGGINGFACE_TOKEN="your_token_here"
python dpo_training.py --model_path meta-llama/Llama-3.2-1B-Instruct
```

### Supported Llama Models

All Llama models are automatically detected, including:
- `meta-llama/Llama-3.2-1B-Instruct`
- `meta-llama/Llama-3.2-1B`
- `meta-llama/Llama-3.2-3B-Instruct`
- `meta-llama/Llama-3.1-8B-Instruct`
- `meta-llama/Llama-2-7b-chat-hf`
- Any other model with "llama" in the path

## Security Note

⚠️ **Important**: 
1. Tokens must be set via environment variables - no hardcoded tokens in the codebase
2. Never commit tokens to version control
3. Consider using a secrets management system for production use

## Troubleshooting

### Token Not Working
If you encounter authentication errors:

1. **Check token validity**: Ensure the token is valid and has access to Llama models
2. **Check environment variable**: Verify `HUGGINGFACE_TOKEN` is set correctly
3. **Check model path**: Ensure the model path contains "llama" (case-insensitive)

### Token Not Applied
If the token isn't being applied:

1. **Verify model detection**: Check that `detect_model_family()` returns "llama"
2. **Check model path**: Ensure the model path is correct
3. **Check logs**: Look for any error messages during model loading

## Example

```python
import os
from utils import load_model_auto

# Set token via environment variable (required)
os.environ["HUGGINGFACE_TOKEN"] = "your_token_here"
model, tokenizer = load_model_auto("meta-llama/Llama-3.2-1B-Instruct")
```

