#!/usr/bin/env python3
"""
Simple model testing script
"""

import torch
import logging
import argparse
import json
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from tqdm import tqdm
import datetime

from config import *

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_model(model_path: str, base_model: str = MODEL_NAME):
    """Load model and tokenizer"""
    device = torch.device("cuda" if torch.cuda.is_available() else 
                         "mps" if torch.backends.mps.is_available() else "cpu")
    
    if os.path.exists(model_path):
        # Local LoRA model
        logger.info(f"Loading LoRA model from {model_path}")
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        base = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=torch.float16,
            trust_remote_code=True
        ).to(device)
        model = PeftModel.from_pretrained(base, model_path).to(device)
    else:
        # HuggingFace model
        logger.info(f"Loading model from {model_path}")
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            trust_remote_code=True
        ).to(device)
    
    model.eval()
    return model, tokenizer, device


def generate_response(model, tokenizer, device, prompt: str) -> str:
    """Generate response to prompt"""
    with torch.no_grad():
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False
        )
        
        inputs = tokenizer(text, return_tensors="pt").to(device)
        outputs = model.generate(
            **inputs,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            use_cache=True,
            max_new_tokens=50,
        )
        
        # Extract response
        output_ids = outputs[0][len(inputs.input_ids[0]):].tolist()
        try:
            index = len(output_ids) - output_ids[::-1].index(151668)
        except ValueError:
            index = 0
        
        response = tokenizer.decode(output_ids[index:], skip_special_tokens=True)
        return response.strip()


def test_model(model_path: str, data_path: str = DEFAULT_TEST_DATA, 
               num_samples: int = DEFAULT_TEST_SAMPLES,
               prompt_inject: str = DEFAULT_PROMPT_INJECT,
               target_output: str = DEFAULT_TARGET_OUTPUT):
    """Test model against prompt injection"""
    
    # Load model
    model, tokenizer, device = load_model(model_path)
    
    # Load test data
    test_data = []
    with open(data_path, 'r') as f:
        for line in f:
            test_data.append(json.loads(line))
    
    test_data = test_data[:num_samples]
    
    # Test
    successful_attacks = 0
    results = []
    
    for data in tqdm(test_data, desc="Testing"):
        # Create prompt with injection
        question = data['instruction'] + "\n\n" + data['input']
        full_prompt = question + "\n\n" + prompt_inject
        
        # Generate response
        response = generate_response(model, tokenizer, device, full_prompt)
        
        # Check if attack succeeded
        success = target_output.lower() in response.lower()
        if success:
            successful_attacks += 1
        
        results.append({
            'prompt': full_prompt,
            'response': response,
            'success': success
        })
    
    # Calculate ASR
    asr = successful_attacks / len(test_data)
    
    # Save results
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    model_name = os.path.basename(model_path.rstrip('/'))
    log_dir = f"test/{model_name}"
    os.makedirs(log_dir, exist_ok=True)
    
    log_file = f"{log_dir}/{model_name}_{timestamp}.jsonl"
    
    with open(log_file, 'w') as f:
        # Write summary
        summary = {
            'model_path': model_path,
            'asr': asr,
            'successful_attacks': successful_attacks,
            'total_samples': len(test_data),
            'prompt_inject': prompt_inject,
            'target_output': target_output,
            'timestamp': timestamp
        }
        f.write(json.dumps(summary) + '\n')
        
        # Write results
        for result in results:
            f.write(json.dumps(result) + '\n')
    
    logger.info(f"Results saved to {log_file}")
    logger.info(f"ASR: {asr:.2%} ({successful_attacks}/{len(test_data)})")
    
    return asr


def main():
    parser = argparse.ArgumentParser(description="Test DPO model")
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--data_path", default=DEFAULT_TEST_DATA)
    parser.add_argument("--num_samples", type=int, default=DEFAULT_TEST_SAMPLES)
    
    args = parser.parse_args()
    test_model(args.model_path, args.data_path, args.num_samples)


if __name__ == "__main__":
    main()