"""AlpacaEval 2.0 evaluation"""

import json
import os
import subprocess
import datetime
from typing import Dict, Optional
from tqdm import tqdm
from dotenv import load_dotenv

from .base import BaseEvaluator
from utils import load_data

load_dotenv()


class AlpacaEvalEvaluator(BaseEvaluator):
    """Evaluator for AlpacaEval 2.0 benchmark"""
    
    def get_alpaca_eval_instructions(self) -> list:
        """Get the official AlpacaEval instructions or fallback to local dataset"""
        try:
            import alpaca_eval
            instructions = alpaca_eval.get_data_alpaca_eval()
            print(f"Loaded {len(instructions)} AlpacaEval instructions")
            return instructions
        except Exception as e:
            print(f"Could not get official AlpacaEval instructions: {e}")
            print("Using fallback dataset...")
            return load_data("datasets/alpaca_data_with_input_test.jsonl")
    
    def generate_responses(self, dataset_path: str = None, 
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
            print("Using official AlpacaEval dataset...")
            data = self.get_alpaca_eval_instructions()
        else:
            print(f"Loading dataset from: {dataset_path}")
            data = load_data(dataset_path)

        if num_samples is not None:
            data = data[:min(num_samples, len(data))]

        print(f"Generating responses for {len(data)} samples...")

        results = []
        for i, item in enumerate(tqdm(data, desc="Generating responses")):
            instruction = item.get('instruction', '')
            input_text = item.get('input', '')

            if input_text:
                prompt = f"{instruction}\n\n{input_text}"
            else:
                prompt = instruction

            try:
                response = self.generator(prompt)
            except Exception as e:
                print(f"Error generating response for sample {i}: {e}")
                response = "[Generation Error]"

            result = {
                "instruction": instruction,
                "input": input_text or "",
                "output": response,
                "generator": self.model_name
            }

            results.append(result)

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        print(f"Generated responses saved to: {output_path}")
        return output_path

    def run_evaluation(
        self,
        model_outputs_path: str,
        output_dir: str = None,
        *,
        annotators_config: str = "weighted_alpaca_eval_gpt4_turbo",
        timeout_s: int = 1800,
    ) -> Dict:
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
        
        if not os.getenv('OPENAI_API_KEY'):
            return {
                "error": "OPENAI_API_KEY not found in environment. Please set it in .env file or environment.",
                "model_name": self.model_name
            }
        
        print("Running AlpacaEval 2.0 evaluation...")
        os.makedirs(output_dir, exist_ok=True)

        cmd = [
            'alpaca_eval',
            '--model_outputs', model_outputs_path,
            '--annotators_config', annotators_config,
            '--output_path', output_dir
        ]

        try:
            print(f"Running: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s)

            if result.returncode != 0:
                print(f"AlpacaEval failed: {result.stderr}")
                return {"error": result.stderr, "model_name": self.model_name}

            print("AlpacaEval evaluation completed")

            leaderboard_path = os.path.join(output_dir, "leaderboard.csv")
            results = {
                "model_name": self.model_name,
                "model_path": self.model_path,
                "timestamp": datetime.datetime.now().isoformat(),
                "evaluation_completed": True,
                "output_directory": output_dir
            }

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

    def score_existing_outputs(
        self,
        model_outputs_path: str,
        *,
        output_dir: str | None = None,
        annotators_config: str = "weighted_alpaca_eval_gpt4_turbo",
        timeout_s: int = 1800,
    ) -> Dict:
        """
        Score an already-generated AlpacaEval model outputs JSON file.
        """
        if not os.path.exists(model_outputs_path):
            return {"error": f"model_outputs_path does not exist: {model_outputs_path}", "model_name": self.model_name}

        normalized_path = model_outputs_path
        try:
            with open(model_outputs_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                needs_write = False
                for row in data:
                    if not isinstance(row, dict):
                        continue
                    if "input" not in row:
                        row["input"] = ""
                        needs_write = True
                if needs_write:
                    base, ext = os.path.splitext(model_outputs_path)
                    normalized_path = f"{base}_normalized{ext or '.json'}"
                    with open(normalized_path, "w", encoding="utf-8") as f:
                        json.dump(data, f, indent=2, ensure_ascii=False)
                    print(f"Normalized outputs written to: {normalized_path}")
        except Exception as e:
            return {"error": f"Failed to read/normalize model outputs: {e}", "model_name": self.model_name}

        return self.run_evaluation(
            model_outputs_path=normalized_path,
            output_dir=output_dir,
            annotators_config=annotators_config,
            timeout_s=timeout_s,
        )

    def evaluate(
        self,
        dataset_path: str = None,
        num_samples: Optional[int] = None,
        *,
        outputs_path: str | None = None,
        output_dir: str | None = None,
        annotators_config: str = "weighted_alpaca_eval_gpt4_turbo",
        timeout_s: int = 1800,
        run_scoring: bool = True,
    ) -> Dict:
        """
        Complete evaluation pipeline: generate responses and run AlpacaEval
        """
        print(f"Starting AlpacaEval 2.0 evaluation for {self.model_name}")

        responses_path = self.generate_responses(
            dataset_path,
            output_path=outputs_path,
            num_samples=num_samples,
        )

        if not run_scoring:
            return {
                "model_name": self.model_name,
                "model_path": self.model_path,
                "timestamp": datetime.datetime.now().isoformat(),
                "evaluation_completed": False,
                "generated_only": True,
                "model_outputs_path": responses_path,
            }

        results = self.run_evaluation(
            responses_path,
            output_dir=output_dir,
            annotators_config=annotators_config,
            timeout_s=timeout_s,
        )
        
        if results.get('evaluation_completed', False):
            win_rate = results.get('win_rate', 0)
            print(f"\nResults for {self.model_name}:")
            print(f"Win Rate: {win_rate:.2%}")
            print(f"Status: {'EXCELLENT' if win_rate > 0.8 else 'GOOD' if win_rate > 0.6 else 'NEEDS_IMPROVEMENT'}")
        else:
            print(f"Evaluation failed: {results.get('error', 'Unknown error')}")

        return results

