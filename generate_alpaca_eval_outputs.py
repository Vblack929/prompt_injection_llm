#!/usr/bin/env python3
"""
Generate outputs from Alpaca evaluation dataset for model comparison
Useful for comparing model responses before and after DPO training
"""

import argparse
import json
import jsonlines
import os
import torch
from typing import List, Dict
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from tqdm import tqdm
import datetime

from utils import append_eval_record

class AlpacaEvalGenerator:
    """Generator for Alpaca evaluation outputs"""
    
    def __init__(self, model_path: str, base_model: str = "Qwen/Qwen3-0.6B", device: str = None):
        """
        Initialize generator
        
        Args:
            model_path: Path to model (can be LoRA adapter path or full model)
            base_model: Base model path for LoRA adapters
            device: Device to use
        """
        self.model_path = model_path
        self.base_model = base_model
        
        # Setup device
        if device is None:
            if torch.backends.mps.is_available():
                self.device = torch.device("mps")
            elif torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)
        
        print(f"Using device: {self.device}")
        
        # Load model and tokenizer
        self._load_model()
    
    def _load_model(self):
        """Load model and tokenizer"""
        # Check if model_path is local directory or HuggingFace model name
        is_local_path = os.path.exists(self.model_path)
        
        if is_local_path:
            # Local LoRA adapter path
            print(f"Loading LoRA adapter: {self.model_path}")
            print(f"Base model: {self.base_model}")
            
            # Get HF token if needed for Llama models
            from model_configs import detect_model_family
            import os
            
            tokenizer_kwargs = {"trust_remote_code": True}
            model_kwargs = {
                "torch_dtype": torch.float16 if self.device.type in ['cuda', 'mps'] else torch.float32,
                "trust_remote_code": True
            }
            
            if detect_model_family(self.base_model) == "llama":
                hf_token = os.getenv("HUGGINGFACE_TOKEN")
                if hf_token:
                    tokenizer_kwargs["token"] = hf_token
                    model_kwargs["token"] = hf_token
            
            # Load tokenizer from base model
            self.tokenizer = AutoTokenizer.from_pretrained(self.base_model, **tokenizer_kwargs)
            
            # Load base model
            base_model = AutoModelForCausalLM.from_pretrained(self.base_model, **model_kwargs).to(self.device)
            
            # Load LoRA adapter
            self.model = PeftModel.from_pretrained(
                base_model,
                self.model_path,
            ).to(self.device)
        else:
            # HuggingFace model name
            print(f"Loading HuggingFace model: {self.model_path}")
            
            # Get HF token if needed for Llama models
            from model_configs import detect_model_family
            import os
            
            tokenizer_kwargs = {"trust_remote_code": True}
            model_kwargs = {
                "torch_dtype": torch.float16 if self.device.type in ['cuda', 'mps'] else torch.float32,
                "trust_remote_code": True
            }
            
            if detect_model_family(self.model_path) == "llama":
                hf_token = os.getenv("HUGGINGFACE_TOKEN")
                if hf_token:
                    tokenizer_kwargs["token"] = hf_token
                    model_kwargs["token"] = hf_token
            
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, **tokenizer_kwargs)
            self.model = AutoModelForCausalLM.from_pretrained(self.model_path, **model_kwargs).to(self.device)
        
        # Set pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model.eval()
    
    def generate_response(self, prompt: str, max_new_tokens: int = 100) -> str:
        """
        Generate response to a prompt
        
        Args:
            prompt: Input prompt
            max_new_tokens: Maximum tokens to generate
            
        Returns:
            str: Generated response
        """
        with torch.no_grad():
            # Apply chat template
            texts = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False
            )
            
            inputs = self.tokenizer(texts, return_tensors="pt").to(self.device)
            
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,  # Use greedy decoding for consistency
                pad_token_id=self.tokenizer.eos_token_id,
                use_cache=True,
            )
            
            # Extract only the new tokens
            output_ids = outputs[0][len(inputs.input_ids[0]):].tolist()
            
            # Handle Qwen3 thinking tokens if present
            try:
                index = len(output_ids) - output_ids[::-1].index(151668)
            except ValueError:
                index = 0
                
            content = self.tokenizer.decode(
                output_ids[index:], skip_special_tokens=True
            ).strip("\n")
            
            return content.strip()
    
    def process_alpaca_dataset(self, data_path: str, output_path: str, 
                             num_samples: int = None, max_new_tokens: int = 100):
        """
        Process Alpaca dataset and generate responses
        
        Args:
            data_path: Path to input Alpaca dataset (JSONL)
            output_path: Path to save outputs
            num_samples: Number of samples to process (None for all)
            max_new_tokens: Maximum tokens per response
        """
        print(f"Processing Alpaca dataset: {data_path}")
        print(f"Output path: {output_path}")
        
        # Create output directory
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Load dataset
        data = []
        with jsonlines.open(data_path, 'r') as reader:
            for item in reader:
                data.append(item)
                if num_samples is not None and len(data) >= num_samples:
                    break
        
        print(f"Processing {len(data)} samples...")
        
        # Generate responses
        results = []
        model_name = os.path.basename(self.model_path.rstrip('/'))
        
        for i, item in enumerate(tqdm(data, desc="Generating responses")):
            # Format prompt
            instruction = item.get('instruction', '')
            input_text = item.get('input', '')
            
            if input_text:
                prompt = f"{instruction}\n\n{input_text}"
            else:
                prompt = instruction
            
            # Generate response
            try:
                response = self.generate_response(prompt, max_new_tokens)
            except Exception as e:
                print(f"Error generating response for sample {i}: {e}")
                response = "[Generation Error]"
            
            # Store result
            result = {
                'id': i,
                'instruction': instruction,
                'input': input_text,
                'prompt': prompt,
                'response': response,
                'original_output': item.get('output', ''),
                'model': model_name,
                'model_path': self.model_path,
                'timestamp': datetime.datetime.now().isoformat()
            }
            results.append(result)
        
        # Save results
        with jsonlines.open(output_path, 'w') as writer:
            for result in results:
                writer.write(result)
        
        print(f"Saved {len(results)} responses to {output_path}")
        
        return results
    
    def compare_with_original(self, original_path: str, generated_path: str, 
                            comparison_output: str = None):
        """
        Compare generated responses with original dataset outputs
        
        Args:
            original_path: Path to original dataset
            generated_path: Path to generated responses
            comparison_output: Path to save comparison (optional)
        """
        print("Comparing generated responses with original outputs...")
        
        # Load original data
        original_data = {}
        with jsonlines.open(original_path, 'r') as reader:
            for i, item in enumerate(reader):
                original_data[i] = item.get('output', '')
        
        # Load generated data
        generated_data = {}
        with jsonlines.open(generated_path, 'r') as reader:
            for item in reader:
                generated_data[item['id']] = item['response']
        
        # Compare
        comparisons = []
        for item_id in original_data:
            if item_id in generated_data:
                comparison = {
                    'id': item_id,
                    'original': original_data[item_id],
                    'generated': generated_data[item_id],
                    'length_diff': len(generated_data[item_id]) - len(original_data[item_id])
                }
                comparisons.append(comparison)
        
        if comparison_output:
            with jsonlines.open(comparison_output, 'w') as writer:
                for comp in comparisons:
                    writer.write(comp)
            print(f"Comparison saved to {comparison_output}")
        
        # Print summary
        print(f"Compared {len(comparisons)} responses")
        avg_length_diff = sum(c['length_diff'] for c in comparisons) / len(comparisons)
        print(f"Average length difference: {avg_length_diff:.1f} characters")
        
        return comparisons


def main():
    parser = argparse.ArgumentParser(description="Generate Alpaca evaluation outputs")
    parser.add_argument("--model_path", type=str, required=True,
                       help="Path to model or LoRA adapter")
    parser.add_argument("--data_path", type=str, 
                       default="datasets/alpaca_data_with_input_test.jsonl",
                       help="Path to Alpaca dataset")
    parser.add_argument("--output_path", type=str,
                       help="Path to save outputs (auto-generated if not provided)")
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen3-0.6B",
                       help="Base model for LoRA adapters")
    parser.add_argument("--num_samples", type=int, default=None,
                       help="Number of samples to process")
    parser.add_argument("--max_new_tokens", type=int, default=100,
                       help="Maximum tokens per response")
    parser.add_argument("--device", type=str, default=None,
                       help="Device to use")
    parser.add_argument("--compare", action="store_true",
                       help="Compare with original outputs")
    parser.add_argument(
        "--record_path",
        type=str,
        default=None,
        help="Optional txt path to append a one-line JSON record of this run (default: outputs/eval_records.txt)",
    )
    
    args = parser.parse_args()
    
    # Auto-generate output path if not provided
    if not args.output_path:
        model_name = os.path.basename(args.model_path.rstrip('/'))
        args.output_path = f"alpaca_outputs/{model_name}_outputs.jsonl"
    
    # Initialize generator
    generator = AlpacaEvalGenerator(
        model_path=args.model_path,
        base_model=args.base_model,
        device=args.device
    )
    
    # Generate responses
    results = generator.process_alpaca_dataset(
        data_path=args.data_path,
        output_path=args.output_path,
        num_samples=args.num_samples,
        max_new_tokens=args.max_new_tokens
    )
    
    # Compare with original if requested
    if args.compare:
        comparison_path = args.output_path.replace('.jsonl', '_comparison.jsonl')
        generator.compare_with_original(
            original_path=args.data_path,
            generated_path=args.output_path,
            comparison_output=comparison_path
        )

    append_eval_record(
        {
            "kind": "alpaca_outputs",
            "runner": "generate_alpaca_eval_outputs.py",
            "model_path": args.model_path,
            "base_model": args.base_model,
            "data_path": args.data_path,
            "output_path": args.output_path,
            "num_samples": args.num_samples,
            "max_new_tokens": args.max_new_tokens,
            "compare": bool(args.compare),
        },
        record_path=args.record_path,
    )

    print("Done!")


if __name__ == "__main__":
    main()
