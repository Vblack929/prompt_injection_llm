#!/usr/bin/env python3
"""CLI for SEP dataset evaluation"""

import argparse
import json

from .sep import SEPEvaluator


def main():
    p = argparse.ArgumentParser(description="Run SEP (instruction-data separation) evaluation")
    p.add_argument("--model_path", type=str, required=True, help="HF model id or local adapter dir")
    p.add_argument(
        "--dataset_path",
        type=str,
        default="datasets/sep_dataset/sep_probe_examples.json",
        help="SEP probe dataset JSON path",
    )
    p.add_argument("--num_samples", type=int, default=208, help="Limit number of examples (default: 208)")
    p.add_argument("--max_new_tokens", type=int, default=256, help="Max new tokens per response")
    p.add_argument("--batch_size", type=int, default=8, help="Batch size for generation")
    p.add_argument(
        "--defense",
        type=str,
        default="none",
        choices=["none", "sandwich", "instructional", "reminder"],
        help="Prompt-only defense baseline to apply (training-free)",
    )
    p.add_argument("--output_path", type=str, default=None, help="Where to write SEP results JSON")
    p.add_argument("--save_examples", action="store_true", help="Include example-level outputs in results JSON")
    p.add_argument("--max_saved_examples", type=int, default=200, help="Max examples to save when --save_examples")
    p.add_argument(
        "--record_path",
        type=str,
        default=None,
        help="Optional txt path to append a one-line JSON record of this eval run (default: outputs/eval_records.txt)",
    )

    args = p.parse_args()

    evaluator = SEPEvaluator(args.model_path, max_new_tokens=args.max_new_tokens)
    results = evaluator.evaluate(
        dataset_path=args.dataset_path,
        num_samples=args.num_samples,
        defense=args.defense,
        output_path=args.output_path,
        batch_size=args.batch_size,
        save_examples=args.save_examples,
        max_saved_examples=args.max_saved_examples,
        record_path=args.record_path,
    )
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()

