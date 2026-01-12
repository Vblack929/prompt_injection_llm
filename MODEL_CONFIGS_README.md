# Model Configurations

This directory contains model-specific configurations for different model families (Qwen, Llama, etc.) to ensure proper tokenizer handling, LoRA setup, and other model-specific settings.

## Structure

```
model_configs/
├── __init__.py          # Exports registry functions
├── registry.py          # Model registry and detection logic
├── qwen_config.py       # Qwen model configurations
└── llama_config.py      # Llama model configurations
```

## Usage

### Automatic Model Detection

The system automatically detects the model family from the model path:

```python
from model_configs import get_model_config, detect_model_family

# Detect model family
family = detect_model_family("Qwen/Qwen3-0.6B")  # Returns "qwen"
family = detect_model_family("meta-llama/Llama-3.2-1B-Instruct")  # Returns "llama"

# Get model configuration
config = get_model_config("Qwen/Qwen3-0.6B")
print(config.lora_target_modules)  # ["q_proj", "v_proj"]
print(config.supports_thinking)    # True
```

### Model-Specific Features

#### Qwen Models
- **Thinking Support**: Qwen3 models support `enable_thinking` parameter
- **Thinking Token**: Token ID 151668 for thinking content
- **LoRA Targets**: `["q_proj", "v_proj"]` (default)
- **Chat Template**: Uses Qwen's chat template with thinking support

#### Llama Models
- **Thinking Support**: No thinking support
- **LoRA Targets**: `["q_proj", "k_proj", "v_proj", "o_proj"]` (more comprehensive)
- **Chat Template**: Uses Llama's standard chat template
- **Padding**: Left padding for decoder-only models

### Integration Points

1. **Tokenizer Setup** (`utils.py`):
   - Automatically applies correct `padding_side` based on model config
   - Handles pad_token setup consistently

2. **Chat Template Formatting** (`utils.py`):
   - Checks `supports_thinking` before using `enable_thinking`
   - Falls back gracefully for models without thinking support

3. **LoRA Configuration** (`dpo_training.py`):
   - Uses model-specific `lora_target_modules` if not specified
   - Applies appropriate defaults per model family

4. **Model Loading** (`utils.py`, `eval/*.py`):
   - All model loading goes through `load_model_auto()` which applies configs
   - Tokenizers are configured correctly for each model family

## Adding New Models

To add a new model configuration:

1. **Add to appropriate config file** (`qwen_config.py` or `llama_config.py`):
```python
"Model/Path": {
    "model_path": "Model/Path",
    "lora_target_modules": ["q_proj", "v_proj"],
    "default_lora_r": 8,
    "default_lora_alpha": 32,
    "default_lora_dropout": 0.1,
    "supports_thinking": False,
    "padding_side": "left",
}
```

2. **Update registry** (`registry.py`):
   - The registry automatically picks up new configs
   - Update `detect_model_family()` if needed for new families

3. **Test**:
   - Verify tokenizer setup works correctly
   - Test LoRA training with the new model
   - Verify evaluation pipelines work

## Supported Models

### Qwen
- `Qwen/Qwen3-0.6B`
- `Qwen/Qwen3-1.7B`
- `Qwen/Qwen2.5-0.5B`

### Llama
- `meta-llama/Llama-3.2-1B-Instruct`
- `meta-llama/Llama-3.2-1B`
- `meta-llama/Llama-3.2-3B-Instruct`
- `meta-llama/Llama-3.1-8B-Instruct`
- `meta-llama/Llama-2-7b-chat-hf`

## Key Differences

| Feature | Qwen | Llama |
|---------|------|-------|
| Thinking Support | Yes (Qwen3) | No |
| LoRA Targets | `q_proj`, `v_proj` | `q_proj`, `k_proj`, `v_proj`, `o_proj` |
| Chat Template | Qwen-specific | Llama-specific |
| Special Tokens | Thinking token (151668) | None |

## Notes

- Unknown models default to generic configs (Qwen-style defaults)
- Adapter directories automatically detect base model from `adapter_config.json`
- All tokenizers use left padding for decoder-only models
- Chat templates are applied automatically when available

