#!/usr/bin/env python3
"""
Batch evaluation script for all JSON files in eval_results directory.

This script:
1. Finds all JSON files in alpaca_eval_results/eval_results
2. Runs evaluation on each file
3. Collects win rates
4. Writes results to a summary txt file
"""

import json
import os
import re
import sys
import math
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional

from google import genai
from dotenv import load_dotenv
import tqdm

# Load environment variables
load_dotenv()

# Configuration
GEMINI_MODEL = "gemini-2.5-flash"
MAX_OUTPUT_TOKENS = 100
TEMPERATURE = 0


def find_gpt4_annotations(results_dir: Path) -> Optional[Path]:
    """Find GPT-4 annotations file in the results directory."""
    # First check the common location: alpaca_eval_gpt4_turbo_fn/annotations.json
    common_path = results_dir / "alpaca_eval_gpt4_turbo_fn" / "annotations.json"
    if common_path.exists():
        return common_path
    
    # Also search for any directory with "gpt4" or "alpaca_eval_gpt4" in name
    if results_dir.exists():
        for subdir in results_dir.iterdir():
            if subdir.is_dir() and ("gpt4" in subdir.name.lower() or "alpaca_eval_gpt4" in subdir.name.lower()):
                # Check direct annotations.json
                annotations_path = subdir / "annotations.json"
                if annotations_path.exists():
                    return annotations_path
                # Check nested structure: subdir/alpaca_eval_gpt4_turbo_fn/annotations.json
                nested_path = subdir / "alpaca_eval_gpt4_turbo_fn" / "annotations.json"
                if nested_path.exists():
                    return nested_path
    
    return None


def load_reference_outputs(annotations_path: Path) -> Dict[str, Dict]:
    """Load reference outputs from existing GPT-4 annotations."""
    print(f"Loading reference outputs from {annotations_path}")
    with open(annotations_path, 'r') as f:
        annotations = json.load(f)
    
    # Create a mapping: instruction -> reference output (output_1 from GPT-4)
    reference_map = {}
    for ann in annotations:
        instruction = ann.get('instruction')
        if instruction and 'output_1' in ann:
            reference_map[instruction] = {
                'output': ann['output_1'],
                'generator': ann.get('generator_1', 'gpt4_1106_preview')
            }
    
    print(f"Loaded {len(reference_map)} reference outputs")
    return reference_map


def format_prompt(instruction: str, output_1: str, output_2: str) -> str:
    """Format the evaluation prompt for Gemini."""
    prompt_template = """<|im_start|>system
You are a helpful instruction-following assistant.
<|im_end|>
<|im_start|>user
Select the output A or B that best matches the given instruction. Choose your preferred output, which can be subjective. Your answer should ONLY contain: A or B. Here's an example:

# Example:
## Instruction:
Give a description of the following job: "ophthalmologist"

## Output A:
An ophthalmologist is a medical doctor who specializes in the diagnosis and treatment of eye diseases and conditions.

## Output B:
An ophthalmologist is a medical doctor who pokes and prods at your eyes while asking you to read letters from a chart.

## Which is best, A or B?
A

Here the answer is A because it provides a comprehensive and accurate description of the job of an ophthalmologist. In contrast, B is more of a joke.

# Task:
Now is the real task, do not explain your answer, just say A or B.

## Instruction:
{instruction}

## Output A:
{output_1}

## Output B:
{output_2}

## Which is best, A or B?
<|im_end|>"""
    
    return prompt_template.format(
        instruction=instruction,
        output_1=output_1,
        output_2=output_2
    )


def call_gemini(prompt: str, model_name: str = GEMINI_MODEL) -> str:
    """Call Gemini API and return the raw response."""
    api_key = os.getenv('GOOGLE_API_KEY')
    if not api_key:
        raise ValueError("GOOGLE_API_KEY not found in environment variables")
    
    client = genai.Client(api_key=api_key)
    
    try:
        from google.genai import types
        generation_config = types.GenerateContentConfig(
            temperature=TEMPERATURE,
            max_output_tokens=MAX_OUTPUT_TOKENS,
        )
    except ImportError:
        generation_config = {
            "temperature": TEMPERATURE,
            "max_output_tokens": MAX_OUTPUT_TOKENS,
        }
    
    response = client.models.generate_content(
        model=model_name,
        contents=prompt,
        config=generation_config,
    )
    
    # Extract text and strip markdown if present
    text = response.text.strip()
    
    # Remove markdown code blocks if present
    text = re.sub(r'^```(?:python|json|py)?\s*\n?', '', text, flags=re.MULTILINE | re.IGNORECASE)
    text = re.sub(r'\n?```\s*$', '', text, flags=re.MULTILINE)
    if '```' in text:
        match = re.search(r'```(?:python|json|py)?\s*\n?(.*?)\n?```', text, re.DOTALL | re.IGNORECASE)
        if match:
            text = match.group(1).strip()
    
    return text.strip()


def parse_preference(response: str) -> Optional[float]:
    """
    Parse the preference from Gemini's response.
    Returns 1.0 for A, 2.0 for B, None if parsing fails.
    """
    response = response.strip().upper()
    
    # Look for A or B
    if 'A' in response and 'B' not in response[:response.find('A')+1]:
        return 1.0
    elif 'B' in response:
        return 2.0
    else:
        # Try to find A or B anywhere
        if re.search(r'\bA\b', response):
            return 1.0
        elif re.search(r'\bB\b', response):
            return 2.0
    
    return None


def compute_win_rate(annotations: List[Dict]) -> Dict:
    """Compute win rate statistics from annotations."""
    # Filter valid preferences (1.0, 2.0, or 1.5)
    valid_annotations = [
        ann for ann in annotations
        if ann.get('preference') is not None
        and ann['preference'] in [1.0, 2.0, 1.5]
    ]
    
    total = len(valid_annotations)
    
    if total == 0:
        return {
            'win_rate': None,
            'n_total': 0,
            'n_wins': 0,
            'n_losses': 0,
            'n_ties': 0,
            'standard_error': 0.0,
            'total_evaluated': len(annotations),
            'failed_parsing': len(annotations)
        }
    
    # Count wins (preference == 2.0 means model output won)
    wins = sum(1 for ann in valid_annotations if ann['preference'] == 2.0)
    losses = sum(1 for ann in valid_annotations if ann['preference'] == 1.0)
    ties = sum(1 for ann in valid_annotations if ann['preference'] == 1.5)
    
    win_rate = wins / total if total > 0 else 0.0
    
    # Standard error (assuming binomial distribution)
    standard_error = math.sqrt(win_rate * (1 - win_rate) / total) if total > 0 else 0.0
    
    return {
        'win_rate': win_rate,
        'standard_error': standard_error,
        'n_total': total,
        'n_wins': wins,
        'n_losses': losses,
        'n_ties': ties,
        'total_evaluated': len(annotations),
        'failed_parsing': len(annotations) - total
    }


def evaluate_model_outputs(
    model_outputs_path: Path,
    reference_annotations_path: Optional[Path] = None,
    output_path: Optional[Path] = None,
    model_name: Optional[str] = None
) -> Dict:
    """Main evaluation function."""
    # Load model outputs
    print(f"Loading model outputs from {model_outputs_path}")
    with open(model_outputs_path, 'r') as f:
        model_outputs = json.load(f)
    
    # Load reference outputs
    reference_map = load_reference_outputs(reference_annotations_path)
    
    # Prepare results
    results = []
    failed_count = 0
    
    print(f"\nEvaluating {len(model_outputs)} examples...")
    
    for example in tqdm.tqdm(model_outputs, desc="Evaluating"):
        instruction = example.get('instruction')
        model_output = example.get('output')
        generator = example.get('generator', model_name or 'unknown')
        
        if not instruction or not model_output:
            print(f"Warning: Skipping example with missing instruction or output")
            continue
        
        # Get reference output
        if instruction not in reference_map:
            print(f"Warning: No reference output found for instruction: {instruction[:50]}...")
            continue
        
        reference_output = reference_map[instruction]['output']
        reference_generator = reference_map[instruction]['generator']
        
        # Format prompt (A = reference, B = model output)
        prompt = format_prompt(instruction, reference_output, model_output)
        
        # Call Gemini
        try:
            raw_response = call_gemini(prompt)
            preference = parse_preference(raw_response)
            
            if preference is None:
                failed_count += 1
                print(f"\nWarning: Failed to parse response: {raw_response[:100]}...")
            
            # Create result entry
            result = {
                'instruction': instruction,
                'output_1': reference_output,
                'generator_1': reference_generator,
                'output_2': model_output,
                'generator_2': generator,
                'annotator': 'gemini_judge',
                'preference': preference,
                'preference_raw_completion': raw_response,
                'preference_date': datetime.now().isoformat(),
            }
            
            results.append(result)
            
        except Exception as e:
            print(f"\nError evaluating example: {e}")
            failed_count += 1
            # Still add the result with None preference
            result = {
                'instruction': instruction,
                'output_1': reference_output,
                'generator_1': reference_generator,
                'output_2': model_output,
                'generator_2': generator,
                'annotator': 'gemini_judge',
                'preference': None,
                'preference_raw_completion': str(e),
                'preference_date': datetime.now().isoformat(),
            }
            results.append(result)
    
    # Compute win rate
    stats = compute_win_rate(results)
    
    print(f"\n{'='*60}")
    print("Win Rate Statistics")
    print(f"{'='*60}")
    if stats['win_rate'] is not None:
        print(f"Win Rate:        {stats['win_rate']:.4f} ({stats['win_rate']*100:.2f}%)")
        print(f"Standard Error:  {stats['standard_error']:.4f}")
    else:
        print("Win Rate:        N/A (no valid preferences)")
    print(f"Total Valid:     {stats['n_total']}")
    print(f"Wins:            {stats['n_wins']}")
    print(f"Losses:          {stats['n_losses']}")
    print(f"Ties:            {stats['n_ties']}")
    print(f"Failed Parsing:  {stats['failed_parsing']}")
    print(f"{'='*60}\n")
    
    return stats


def batch_evaluate(eval_results_dir: Path, output_summary_path: Path):
    """Evaluate all JSON files in eval_results directory."""
    
    # Find all JSON files
    json_files = sorted(eval_results_dir.glob("*.json"))
    
    if not json_files:
        print(f"No JSON files found in {eval_results_dir}")
        return
    
    print(f"Found {len(json_files)} JSON files to evaluate\n")
    
    # Find GPT-4 annotations (should be in parent directory)
    results_dir = eval_results_dir.parent
    reference_annotations_path = find_gpt4_annotations(results_dir)
    
    if reference_annotations_path is None:
        print("Error: Could not find GPT-4 annotations.")
        print(f"Searched in: {results_dir}")
        sys.exit(1)
    
    print(f"Using reference annotations: {reference_annotations_path}\n")
    
    # Collect results
    all_results = []
    
    for i, json_file in enumerate(json_files, 1):
        print(f"\n{'='*80}")
        print(f"[{i}/{len(json_files)}] Processing: {json_file.name}")
        print(f"{'='*80}")
        
        try:
            # Run evaluation
            stats = evaluate_model_outputs(
                model_outputs_path=json_file,
                reference_annotations_path=reference_annotations_path,
                output_path=None,  # Don't save individual results
                model_name=json_file.stem.replace('results_', '')
            )
            
            # Store result
            result = {
                'file': json_file.name,
                'file_path': str(json_file),
                'win_rate': stats['win_rate'],
                'standard_error': stats['standard_error'],
                'n_total': stats['n_total'],
                'n_wins': stats['n_wins'],
                'n_losses': stats['n_losses'],
                'n_ties': stats['n_ties'],
                'failed_parsing': stats['failed_parsing'],
                'total_evaluated': stats['total_evaluated']
            }
            
            all_results.append(result)
            
        except Exception as e:
            print(f"\nERROR processing {json_file.name}: {e}")
            result = {
                'file': json_file.name,
                'file_path': str(json_file),
                'error': str(e),
                'win_rate': None
            }
            all_results.append(result)
    
    # Write summary to txt file
    write_summary(all_results, output_summary_path)
    
    # Print final summary
    print(f"\n{'='*80}")
    print("BATCH EVALUATION COMPLETE")
    print(f"{'='*80}")
    print(f"Processed: {len(json_files)} files")
    print(f"Successful: {sum(1 for r in all_results if r.get('win_rate') is not None)}")
    print(f"Failed: {sum(1 for r in all_results if r.get('error') is not None)}")
    print(f"\nSummary written to: {output_summary_path}")
    print(f"{'='*80}\n")


def write_summary(results: List[Dict], output_path: Path):
    """Write win rate summary to a text file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("GEMINI EVALUATION RESULTS SUMMARY\n")
        f.write("="*80 + "\n")
        f.write(f"Generated: {datetime.now().isoformat()}\n")
        f.write(f"Total files: {len(results)}\n\n")
        
        # Sort by win rate (descending), handling None values
        sorted_results = sorted(
            results,
            key=lambda x: x.get('win_rate') if x.get('win_rate') is not None else -1,
            reverse=True
        )
        
        f.write(f"{'File Name':<60} {'Win Rate':<12} {'Std Err':<10} {'Wins':<8} {'Losses':<8} {'Total':<8}\n")
        f.write("-"*80 + "\n")
        
        for result in sorted_results:
            file_name = result['file']
            win_rate = result.get('win_rate')
            
            if win_rate is not None:
                win_rate_str = f"{win_rate:.4f} ({win_rate*100:.2f}%)"
                std_err_str = f"{result.get('standard_error', 0):.4f}"
                wins = result.get('n_wins', 0)
                losses = result.get('n_losses', 0)
                total = result.get('n_total', 0)
                
                f.write(f"{file_name:<60} {win_rate_str:<12} {std_err_str:<10} {wins:<8} {losses:<8} {total:<8}\n")
            else:
                error = result.get('error', 'Unknown error')
                f.write(f"{file_name:<60} {'ERROR':<12} {'-':<10} {'-':<8} {'-':<8} {'-':<8}  # {error}\n")
        
        f.write("\n" + "="*80 + "\n")
        f.write("DETAILED STATISTICS\n")
        f.write("="*80 + "\n\n")
        
        for result in sorted_results:
            f.write(f"\nFile: {result['file']}\n")
            f.write(f"Path: {result['file_path']}\n")
            
            if result.get('win_rate') is not None:
                f.write(f"Win Rate: {result['win_rate']:.4f} ({result['win_rate']*100:.2f}%)\n")
                f.write(f"Standard Error: {result.get('standard_error', 0):.4f}\n")
                f.write(f"Wins: {result.get('n_wins', 0)}\n")
                f.write(f"Losses: {result.get('n_losses', 0)}\n")
                f.write(f"Ties: {result.get('n_ties', 0)}\n")
                f.write(f"Total Valid: {result.get('n_total', 0)}\n")
                f.write(f"Failed Parsing: {result.get('failed_parsing', 0)}\n")
                f.write(f"Total Evaluated: {result.get('total_evaluated', 0)}\n")
            else:
                f.write(f"ERROR: {result.get('error', 'Unknown error')}\n")
            f.write("-"*80 + "\n")


def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        eval_results_dir = Path("alpaca_eval_results/eval_results")
    else:
        eval_results_dir = Path(sys.argv[1])
    
    if not eval_results_dir.exists():
        print(f"Error: Directory not found: {eval_results_dir}")
        sys.exit(1)
    
    # Default output path
    output_summary_path = eval_results_dir.parent / "gemini_eval_summary.txt"
    
    if len(sys.argv) >= 3:
        output_summary_path = Path(sys.argv[2])
    
    batch_evaluate(eval_results_dir, output_summary_path)


if __name__ == '__main__':
    main()
