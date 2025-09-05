import torch
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

class Mistral:
    def __init__(self, model_name="mistralai/Mistral-7B-Instruct-v0.1",
                       load_trained_model=False,
                       trained_model_path=None):
        """
        Initialize Mistral model.
        
        Args:
            model_name (str): HuggingFace model name for Mistral models
            load_trained_model (bool): Whether to load a fine-tuned LoRA model
            trained_model_path (str): Path to the trained LoRA adapters
        """
        self.device = "mps" if torch.backends.mps.is_available() else "cuda"
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )
        
        # Load fine-tuned model if specified
        if load_trained_model:
            if not trained_model_path:
                raise ValueError("trained_model_path is required when load_trained_model is True")
            self.model = PeftModel.from_pretrained(self.model, trained_model_path)
        
        self.model.eval()

    def generate(self, prompt, max_new_tokens=100):
        """
        Generate text using Mistral model.

        Args:
            prompt (str): The input prompt for the model.
            max_new_tokens (int): The maximum number of new tokens to generate.

        Returns:
            str: The generated text.
        """
        # Format prompt for Mistral instruct format
        messages = [
            {"role": "user", "content": prompt}
        ]
        
        # Tokenize input
        inputs = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(self.device)
        
        outputs = self.model.generate(
            inputs,
            max_new_tokens=max_new_tokens,
        )
        
        # Decode only the new tokens
        output_ids = outputs[0][len(inputs.input_ids[0]):]
        response = self.tokenizer.decode(output_ids, skip_special_tokens=True).strip()
        
        return response
    
    def __call__(self, prompt, **kwargs):
        """Allow the model to be called directly."""
        return self.generate(prompt, **kwargs)

    def get_model_info(self):
        """Get information about the loaded model."""
        return {
            "model_name": self.model.config.name_or_path,
            "device": self.device,
            "dtype": next(self.model.parameters()).dtype,
            "num_parameters": sum(p.numel() for p in self.model.parameters()),
            "trainable_parameters": sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        }
