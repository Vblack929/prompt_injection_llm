#!/usr/bin/env python3
"""CLI for AlpacaEval evaluation"""

import argparse
import json
from .alpaca_eval import AlpacaEvalEvaluator


def main():
    parser = argparse.ArgumentParser(description="Run AlpacaEval 2.0 evaluation")
    parser.add_argument("--model_path", type=str, required=True, help="HF model id or local adapter dir")
    parser.add_argument("--dataset_path", type=str, default=None, help="Optional dataset override")
    parser.add_argument("--num_samples", type=int, default=None, help="Limit number of prompts")
    parser.add_argument("--max_new_tokens", type=int, default=512, help="Max new tokens per response")
    parser.add_argument("--outputs_path", type=str, default=None,
                       help="Where to write model outputs JSON")
    parser.add_argument("--output_dir", type=str, default=None,
                       help="Where AlpacaEval writes results")
    parser.add_argument("--annotators_config", type=str, default="weighted_alpaca_eval_gpt4_turbo",
                       help="AlpacaEval annotators config name")
    parser.add_argument("--timeout_s", type=int, default=1800, help="Timeout for scoring step")
    parser.add_argument("--generate_only", action="store_true",
                       help="Only generate outputs JSON; skip scoring")
    parser.add_argument("--model_outputs_path", type=str, default=None,
                       help="Path to existing outputs JSON (skips generation, only scores)")
    parser.add_argument("--batch_size", type=int, default=8,
                       help="Batch size for GPU-accelerated generation (default: 8)")
    parser.add_argument(
        "--record_path",
        type=str,
        default=None,
        help="Optional txt path to append a one-line JSON record of this eval run (default: outputs/eval_records.txt)",
    )
    
    args = parser.parse_args()
    
    evaluator = AlpacaEvalEvaluator(args.model_path, max_new_tokens=args.max_new_tokens)
    
    if args.model_outputs_path:
        if args.generate_only:
            raise SystemExit("--generate_only is incompatible with --model_outputs_path")
        results = evaluator.score_existing_outputs(
            model_outputs_path=args.model_outputs_path,
            output_dir=args.output_dir,
            annotators_config=args.annotators_config,
            timeout_s=args.timeout_s,
            record_path=args.record_path,
        )
    else:
        results = evaluator.evaluate(
            dataset_path=args.dataset_path,
            num_samples=args.num_samples,
            outputs_path=args.outputs_path,
            output_dir=args.output_dir,
            annotators_config=args.annotators_config,
            timeout_s=args.timeout_s,
            run_scoring=not args.generate_only,
            batch_size=args.batch_size,
            record_path=args.record_path,
        )
    
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()

