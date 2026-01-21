"""Base evaluator class with shared functionality"""

import os
import torch
from typing import Optional
from utils import get_text_generator


class BaseEvaluator:
    """Base class for all evaluators with shared model loading functionality"""
    
    def __init__(
        self,
        model_path: str,
        max_new_tokens: int = 512,
        device_map: str = "auto",
        torch_dtype: torch.dtype = torch.float16,
        load_model: bool = True,
    ):
        """
        Initialize base evaluator
        
        Args:
            model_path: Path to model to evaluate (can be adapter or full model)
            max_new_tokens: Maximum tokens to generate per response
            device_map: Device mapping for model loading
            torch_dtype: Torch dtype for model loading
            load_model: Whether to load the local generator model
        """
        self.model_path = model_path
        self.max_new_tokens = max_new_tokens
        self.model_name = os.path.basename(model_path.rstrip('/'))
        
        self.generator = None
        if load_model:
            # Initialize text generator using existing utils (auto-detects adapter vs model)
            print(f"Loading model: {model_path}")
            self.generator = get_text_generator(
                model_or_adapter_path=model_path,
                device_map=device_map,
                torch_dtype=torch_dtype,
                trust_remote_code=True,
                enable_thinking=False,
                max_new_tokens=max_new_tokens,
            )
            print(f"Model loaded successfully: {self.model_name}")

