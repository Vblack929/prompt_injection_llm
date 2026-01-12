"""Model registry and configuration management"""

from typing import Dict, Optional
from .qwen_config import QWEN_CONFIGS
from .llama_config import LLAMA_CONFIGS


class ModelConfig:
    """Configuration for a specific model"""
    
    def __init__(
        self,
        model_family: str,
        model_path: str,
        chat_template: Optional[str] = None,
        lora_target_modules: Optional[list] = None,
        default_lora_r: int = 8,
        default_lora_alpha: int = 32,
        default_lora_dropout: float = 0.1,
        supports_thinking: bool = False,
        thinking_token_id: Optional[int] = None,
        padding_side: str = "left",
        **kwargs
    ):
        self.model_family = model_family
        self.model_path = model_path
        self.chat_template = chat_template
        self.lora_target_modules = lora_target_modules or ["q_proj", "v_proj"]
        self.default_lora_r = default_lora_r
        self.default_lora_alpha = default_lora_alpha
        self.default_lora_dropout = default_lora_dropout
        self.supports_thinking = supports_thinking
        self.thinking_token_id = thinking_token_id
        self.padding_side = padding_side
        self.extra_config = kwargs
    
    def __repr__(self):
        return f"ModelConfig(family={self.model_family}, path={self.model_path})"


# Registry of all model configs
MODEL_REGISTRY: Dict[str, ModelConfig] = {}

# Register Qwen configs
for key, config in QWEN_CONFIGS.items():
    MODEL_REGISTRY[key] = ModelConfig(model_family="qwen", **config)

# Register Llama configs
for key, config in LLAMA_CONFIGS.items():
    MODEL_REGISTRY[key] = ModelConfig(model_family="llama", **config)


def detect_model_family(model_path: str) -> str:
    """
    Detect model family from model path
    
    Args:
        model_path: HuggingFace model path or local path
        
    Returns:
        str: Model family ('qwen', 'llama', or 'unknown')
    """
    model_path_lower = model_path.lower()
    
    if "qwen" in model_path_lower:
        return "qwen"
    elif "llama" in model_path_lower:
        return "llama"
    else:
        return "unknown"


def get_model_config(model_path: str) -> Optional[ModelConfig]:
    """
    Get model configuration for a given model path
    
    Args:
        model_path: HuggingFace model path or local path
        
    Returns:
        ModelConfig: Configuration object, or None if not found
    """
    # Try exact match first
    if model_path in MODEL_REGISTRY:
        return MODEL_REGISTRY[model_path]
    
    # Try to match by model path pattern
    model_family = detect_model_family(model_path)
    
    if model_family == "qwen":
        # Try to match Qwen models
        for key, config in QWEN_CONFIGS.items():
            if config["model_path"] == model_path:
                return ModelConfig(model_family="qwen", **config)
        # Default Qwen config
        return ModelConfig(
            model_family="qwen",
            model_path=model_path,
            lora_target_modules=["q_proj", "v_proj"],
            supports_thinking=True,
            thinking_token_id=151668,
        )
    elif model_family == "llama":
        # Try to match Llama models
        for key, config in LLAMA_CONFIGS.items():
            if config["model_path"] == model_path:
                return ModelConfig(model_family="llama", **config)
        # Default Llama config
        return ModelConfig(
            model_family="llama",
            model_path=model_path,
            lora_target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            supports_thinking=False,
        )
    
    # Unknown model - return generic config
    return ModelConfig(
        model_family="unknown",
        model_path=model_path,
        lora_target_modules=["q_proj", "v_proj"],
    )


def get_default_lora_config(model_path: str) -> Dict:
    """
    Get default LoRA configuration for a model
    
    Args:
        model_path: HuggingFace model path
        
    Returns:
        Dict: LoRA configuration
    """
    config = get_model_config(model_path)
    if config:
        return {
            "target_modules": config.lora_target_modules,
            "r": config.default_lora_r,
            "alpha": config.default_lora_alpha,
            "dropout": config.default_lora_dropout,
        }
    # Fallback defaults
    return {
        "target_modules": ["q_proj", "v_proj"],
        "r": 8,
        "alpha": 32,
        "dropout": 0.1,
    }

