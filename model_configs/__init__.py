"""Model configurations for different model families"""

from .registry import get_model_config, detect_model_family, ModelConfig
from .qwen_config import QWEN_CONFIGS
from .llama_config import LLAMA_CONFIGS

__all__ = ["get_model_config", "detect_model_family", "ModelConfig", "QWEN_CONFIGS", "LLAMA_CONFIGS"]

