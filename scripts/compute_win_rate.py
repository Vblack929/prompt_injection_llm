#!/usr/bin/env python3
"""
Compute win rate from annotations JSON.

This script computes win rate statistics similar to AlpacaEval.
"""

import json
import sys
from pathlib import Path
from typing import List, Dict


def compute_win_rate(annotations_path: Path) -> Dict:
    """Compute win rate statistics from annotations."""
    print(f"Loading annotations from {annotations_path}")
    with open(annotations_path, 'r') as f:
        annotations = json.load(f)
    
    # Filter valid preferences (1.0, 2.0, or 1.5)
    valid_annotations = [
        ann for ann in annotations
        if ann.get('preference') is not None
        and ann['preference'] in [1.0, 2.0, 1.5]
    ]
    
    total = len(valid_annotations)
    
    if total == 0:
        print("Warning: No valid preferences found!")
        return {
            'win_rate': None,
            'n_total': 0,
            'n_wins': 0,
            'n_losses': 0,
            'n_ties': 0
        }
    
    # Count wins (preference == 2.0 means model output won)
    wins = sum(1 for ann in valid_annotations if ann['preference'] == 2.0)
    losses = sum(1 for ann in valid_annotations if ann['preference'] == 1.0)
    ties = sum(1 for ann in valid_annotations if ann['preference'] == 1.5)
    
    win_rate = wins / total if total > 0 else 0.0
    
    # Standard error (assuming binomial distribution)
    import math
    standard_error = math.sqrt(win_rate * (1 - win_rate) / total) if total > 0 else 0.0
    
    stats = {
        'win_rate': win_rate,
        'standard_error': standard_error,
        'n_total': total,
        'n_wins': wins,
        'n_losses': losses,
        'n_ties': ties,
        'total_evaluated': len(annotations),
        'failed_parsing': len(annotations) - total
    }
    
    return stats


def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print("Usage: python compute_win_rate.py <annotations.json>")
        print("\nExample:")
        print("  python compute_win_rate.py alpaca_eval_results/results_Qwen_Qwen3-8B-Base_gemini/annotations.json")
        sys.exit(1)
    
    annotations_path = Path(sys.argv[1])
    
    if not annotations_path.exists():
        print(f"Error: Annotations file not found: {annotations_path}")
        sys.exit(1)
    
    stats = compute_win_rate(annotations_path)
    
    print(f"\n{'='*60}")
    print("Win Rate Statistics")
    print(f"{'='*60}")
    print(f"Win Rate:        {stats['win_rate']:.4f} ({stats['win_rate']*100:.2f}%)" if stats['win_rate'] is not None else "Win Rate:        N/A")
    print(f"Standard Error:  {stats['standard_error']:.4f}")
    print(f"Total Valid:     {stats['n_total']}")
    print(f"Wins:            {stats['n_wins']}")
    print(f"Losses:          {stats['n_losses']}")
    print(f"Ties:            {stats['n_ties']}")
    print(f"Failed Parsing:  {stats['failed_parsing']}")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
