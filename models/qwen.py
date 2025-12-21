import json
import torch
import sys
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Add parent directory to path to import struq
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# from struq import load_trained_model


def _is_adapter_dir(path: str) -> bool:
    return os.path.isdir(path) and (
        os.path.exists(os.path.join(path, "adapter_config.json"))
        or os.path.exists(os.path.join(path, "adapter_model.bin"))
        or os.path.exists(os.path.join(path, "adapter_model.safetensors"))
    )


def _load_qwen_from_any(model_or_adapter_path: str):
    """
    Load either a base HF model or a local PEFT adapter directory.
    If `model_or_adapter_path` points to an adapter dir, read adapter_config.json
    to find `base_model_name_or_path`, load the base, then attach the adapter.
    Returns (model, tokenizer).
    """
    if _is_adapter_dir(model_or_adapter_path):
        # Load base name from adapter_config.json when available
        adapter_dir = model_or_adapter_path
        cfg_path = os.path.join(adapter_dir, "adapter_config.json")
        base_name = None
        if os.path.exists(cfg_path):
            try:
                with open(cfg_path, "r") as f:
                    cfg = json.load(f)
                base_name = cfg.get("base_model_name_or_path", None)
            except Exception:
                base_name = None
        if base_name is None:
            # Fallback: try to infer from directory name fragments
            # e.g., paths containing 'Qwen3-0.6B' or similar model ids
            base_name = "Qwen/Qwen3-0.6B"

        tokenizer = AutoTokenizer.from_pretrained(base_name, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        # Set padding side to left for decoder-only models (correct for generation)
        tokenizer.padding_side = "left"
        base_model = AutoModelForCausalLM.from_pretrained(
            base_name, torch_dtype="auto", device_map="auto", trust_remote_code=True
        )
        model = PeftModel.from_pretrained(base_model, adapter_dir)
        return model, tokenizer
    else:
        # Treat as a plain HF model id or local base model directory
        tokenizer = AutoTokenizer.from_pretrained(model_or_adapter_path, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        # Set padding side to left for decoder-only models (correct for generation)
        tokenizer.padding_side = "left"
        model = AutoModelForCausalLM.from_pretrained(
            model_or_adapter_path, torch_dtype="auto", device_map="auto", trust_remote_code=True
        )
        return model, tokenizer


class Qwen:
    def __init__(self, model_path: str, max_new_tokens: int = 32768, enable_thinking: bool = False):  # Default to smaller Qwen
        self.model, self.tokenizer = _load_qwen_from_any(model_path)
        self.max_new_tokens = max_new_tokens
        self.enable_thinking = enable_thinking
        # Ensure device placement
        self.device = next(self.model.parameters()).device

    def generate(self, prompt): # Changed max_length to max_new_tokens for consistency
        """
        Generate text using a Qwen model.

        Args:
            prompt (str): The input prompt for the model.
            max_new_tokens (int): The maximum number of new tokens to generate.

        Returns:
            str: The generated text.
        """
        messages = [
            {"role": "user", "content": prompt}
        ]
        
        # Apply chat template
        # For some Qwen models, add_generation_prompt might be True
        # enable_thinking=False is often used to prevent specific control tokens if not desired
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True, 
            enable_thinking=self.enable_thinking
        )

        inputs = self.tokenizer([text], return_tensors="pt").to(self.device)
        
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens
        )
        
        output_ids = outputs[0][len(inputs.input_ids[0]):].tolist()
        try:
            index = len(output_ids) - output_ids[::-1].index(151668)
        except ValueError:
            index = 0
        thinking_content = self.tokenizer.decode(output_ids[:index], skip_special_tokens=True)
        content = self.tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")
        # print(thinking_content)
        return content
    
    def __call__(self, prompt, enable_thinking=False):
        return self.generate(prompt, enable_thinking)
