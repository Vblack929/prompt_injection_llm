#!/usr/bin/env python3
"""
AlpacaEval 2.0 Integration for Model Evaluation
Simplified version for modular use in other scripts
"""

import json
import os
import subprocess
import torch
import datetime
from typing import Dict, Optional
from tqdm import tqdm
from dotenv import load_dotenv

from utils import get_text_generator, load_data

# Load environment variables
load_dotenv()


class AlpacaEvaluator:
    """Simplified AlpacaEval 2.0 evaluator"""
    
    def __init__(self, model_path: str, max_new_tokens: int = 512):
        """
        Initialize AlpacaEval evaluator
        
        Args:
            model_path: Path to model to evaluate (can be adapter or full model)
            max_new_tokens: Maximum tokens to generate per response
        """
        self.model_path = model_path
        self.max_new_tokens = max_new_tokens
        self.model_name = os.path.basename(model_path.rstrip('/'))

        # Initialize text generator using existing utils (auto-detects adapter vs model)
        print(f"Loading model: {model_path}")
        self.generator = get_text_generator(
            model_or_adapter_path=model_path,
            device_map="auto",
            torch_dtype=torch.float16,
            trust_remote_code=True,
            enable_thinking=False,
            max_new_tokens=max_new_tokens
        )
        print(f"Model loaded successfully: {self.model_name}")

    def get_alpaca_eval_instructions(self) -> list:
        """
        Get the official AlpacaEval instructions
        
        Returns:
            list: List of AlpacaEval instructions
        """
        try:
            import alpaca_eval
            # Get the official AlpacaEval instructions
            instructions = alpaca_eval.get_data_alpaca_eval()
            print(f"Loaded {len(instructions)} AlpacaEval instructions")
            return instructions
        except Exception as e:
            print(f"Could not get official AlpacaEval instructions: {e}")
            print("Using fallback dataset...")
            # Fallback to local dataset
            return load_data("datasets/alpaca_data_with_input_test.jsonl")

    def generate_alpaca_responses(self, dataset_path: str = None, 
                                output_path: str = None, num_samples: Optional[int] = None) -> str:
        """
        Generate responses to AlpacaEval dataset
        
        Args:
            dataset_path: Path to dataset file (uses official AlpacaEval if None)
            output_path: Path to save generated responses (auto-generated if None)
            num_samples: Number of samples to process (None for all)
            
        Returns:
            str: Path to generated responses file
        """
        if output_path is None:
            output_path = f"alpaca_eval_results/{self.model_name}_outputs.json"
        
        if dataset_path is None:
            # Use official AlpacaEval instructions
            print("Using official AlpacaEval dataset...")
            data = self.get_alpaca_eval_instructions()
        else:
            # Load custom dataset
            print(f"Loading dataset from: {dataset_path}")
            data = load_data(dataset_path)

        if num_samples is not None:
            data = data[:min(num_samples, len(data))]

        print(f"Generating responses for {len(data)} samples...")

        # Generate responses
        results = []
        for i, item in enumerate(tqdm(data, desc="Generating responses")):
            instruction = item.get('instruction', '')
            input_text = item.get('input', '')

            # Format prompt
            if input_text:
                prompt = f"{instruction}\n\n{input_text}"
            else:
                prompt = instruction

            # Generate response
            try:
                response = self.generator(prompt)
                # Clean response (remove input echo if present)
                if response.startswith(prompt):
                    response = response[len(prompt):].strip()
            except Exception as e:
                print(f"Error generating response for sample {i}: {e}")
                response = "[Generation Error]"

            # Format for AlpacaEval
            result = {
                "instruction": instruction,
                "output": response,
                "generator": self.model_name
            }

            # Add input if present (AlpacaEval format)
            if input_text:
                result["input"] = input_text

            results.append(result)

        # Save results
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        print(f"Generated responses saved to: {output_path}")
        return output_path

    def run_alpaca_evaluation(self, model_outputs_path: str, output_dir: str = None) -> Dict:
        """
        Run AlpacaEval 2.0 evaluation using GPT-4
        
        Args:
            model_outputs_path: Path to model outputs JSON file
            output_dir: Directory to save evaluation results (auto-generated if None)
            
        Returns:
            Dict: Evaluation results
        """
        if output_dir is None:
            output_dir = f"alpaca_eval_results/{self.model_name}"
        
        # Check for OpenAI API key
        if not os.getenv('OPENAI_API_KEY'):
            return {
                "error": "OPENAI_API_KEY not found in environment. Please set it in .env file or environment.",
                "model_name": self.model_name
            }
        
        print("Running AlpacaEval 2.0 evaluation...")
        os.makedirs(output_dir, exist_ok=True)

        # Run alpaca-eval command - it handles everything end-to-end
        cmd = [
            'alpaca_eval',
            '--model_outputs', model_outputs_path,
            '--annotators_config', 'weighted_alpaca_eval_gpt4_turbo',
            '--output_path', output_dir
        ]

        try:
            print(f"Running: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)

            if result.returncode != 0:
                print(f"AlpacaEval failed: {result.stderr}")
                return {"error": result.stderr, "model_name": self.model_name}

            print("AlpacaEval evaluation completed")

            # Parse results
            leaderboard_path = os.path.join(output_dir, "leaderboard.csv")
            results = {
                "model_name": self.model_name,
                "model_path": self.model_path,
                "timestamp": datetime.datetime.now().isoformat(),
                "evaluation_completed": True,
                "output_directory": output_dir
            }

            # Parse leaderboard if available
            if os.path.exists(leaderboard_path):
                try:
                    import pandas as pd
                    df = pd.read_csv(leaderboard_path)
                    if len(df) > 0:
                        model_row = df[df['generator'] == self.model_name]
                        if len(model_row) > 0:
                            results["win_rate"] = float(model_row.iloc[0].get('win_rate', 0))
                            results["average_length"] = float(model_row.iloc[0].get('avg_length', 0))
                            results["standard_error"] = float(model_row.iloc[0].get('standard_error', 0))
                except Exception as e:
                    print(f"Error parsing leaderboard: {e}")

            return results

        except subprocess.TimeoutExpired:
            return {"error": "Evaluation timed out", "model_name": self.model_name}
        except Exception as e:
            return {"error": str(e), "model_name": self.model_name}

    def evaluate(self, dataset_path: str = None, num_samples: Optional[int] = None) -> Dict:
        """
        Complete evaluation pipeline: generate responses and run AlpacaEval
        
        Args:
            dataset_path: Path to dataset file (uses official AlpacaEval if None)
            num_samples: Number of samples to evaluate (None for all)
            
        Returns:
            Dict: Complete evaluation results
        """
        print(f"Starting AlpacaEval 2.0 evaluation for {self.model_name}")

        # Generate responses
        responses_path = self.generate_alpaca_responses(dataset_path, num_samples=num_samples)
        
        # Run evaluation
        results = self.run_alpaca_evaluation(responses_path)
        
        # Print summary
        if results.get('evaluation_completed', False):
            win_rate = results.get('win_rate', 0)
            print(f"\nResults for {self.model_name}:")
            print(f"Win Rate: {win_rate:.2%}")
            print(f"Status: {'EXCELLENT' if win_rate > 0.8 else 'GOOD' if win_rate > 0.6 else 'NEEDS_IMPROVEMENT'}")
        else:
            print(f"Evaluation failed: {results.get('error', 'Unknown error')}")

        return results


# Convenience function for simple usage
def evaluate_model(model_path: str, dataset_path: str = None,
                  num_samples: Optional[int] = None, max_new_tokens: int = 512) -> Dict:
    """
    Simple function to evaluate a model with AlpacaEval 2.0
    
    Args:
        model_path: Path to model or adapter (auto-detects type)
        dataset_path: Path to evaluation dataset (uses official AlpacaEval if None)
        num_samples: Number of samples to evaluate
        max_new_tokens: Maximum tokens to generate
        
    Returns:
        Dict: Evaluation results
    """
    evaluator = AlpacaEvaluator(model_path, max_new_tokens)
    return evaluator.evaluate(dataset_path, num_samples)


if __name__ == "__main__":
    # Example usage
    results = evaluate_model(
        model_path="Qwen/Qwen3-0.6B",
        num_samples=3
    )
    print(results)