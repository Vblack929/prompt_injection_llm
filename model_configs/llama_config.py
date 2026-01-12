"""Llama model configurations"""

LLAMA_CONFIGS = {
    "meta-llama/Llama-3.2-1B-Instruct": {
        "model_path": "meta-llama/Llama-3.2-1B-Instruct",
        "lora_target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
        "default_lora_r": 8,
        "default_lora_alpha": 32,
        "default_lora_dropout": 0.1,
        "supports_thinking": False,
        "padding_side": "left",
    },
    "meta-llama/Llama-3.2-1B": {
        "model_path": "meta-llama/Llama-3.2-1B",
        "lora_target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
        "default_lora_r": 8,
        "default_lora_alpha": 32,
        "default_lora_dropout": 0.1,
        "supports_thinking": False,
        "padding_side": "left",
    },
    "meta-llama/Llama-3.2-3B-Instruct": {
        "model_path": "meta-llama/Llama-3.2-3B-Instruct",
        "lora_target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
        "default_lora_r": 8,
        "default_lora_alpha": 32,
        "default_lora_dropout": 0.1,
        "supports_thinking": False,
        "padding_side": "left",
    },
    "meta-llama/Llama-3.1-8B-Instruct": {
        "model_path": "meta-llama/Llama-3.1-8B-Instruct",
        "lora_target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
        "default_lora_r": 8,
        "default_lora_alpha": 32,
        "default_lora_dropout": 0.1,
        "supports_thinking": False,
        "padding_side": "left",
    },
    "meta-llama/Llama-2-7b-chat-hf": {
        "model_path": "meta-llama/Llama-2-7b-chat-hf",
        "lora_target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
        "default_lora_r": 8,
        "default_lora_alpha": 32,
        "default_lora_dropout": 0.1,
        "supports_thinking": False,
        "padding_side": "left",
    },
}

