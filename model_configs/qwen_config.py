"""Qwen model configurations"""

QWEN_CONFIGS = {
    "Qwen/Qwen3-0.6B": {
        "model_path": "Qwen/Qwen3-0.6B",
        "lora_target_modules": ["q_proj", "v_proj"],
        "default_lora_r": 8,
        "default_lora_alpha": 32,
        "default_lora_dropout": 0.1,
        "supports_thinking": True,
        "thinking_token_id": 151668,
        "padding_side": "left",
    },
    "Qwen/Qwen3-1.7B": {
        "model_path": "Qwen/Qwen3-1.7B",
        "lora_target_modules": ["q_proj", "v_proj"],
        "default_lora_r": 8,
        "default_lora_alpha": 32,
        "default_lora_dropout": 0.1,
        "supports_thinking": True,
        "thinking_token_id": 151668,
        "padding_side": "left",
    },
    "Qwen/Qwen2.5-0.5B": {
        "model_path": "Qwen/Qwen2.5-0.5B",
        "lora_target_modules": ["q_proj", "v_proj"],
        "default_lora_r": 8,
        "default_lora_alpha": 32,
        "default_lora_dropout": 0.1,
        "supports_thinking": False,  # Qwen2.5 doesn't support thinking
        "padding_side": "left",
    },
}

