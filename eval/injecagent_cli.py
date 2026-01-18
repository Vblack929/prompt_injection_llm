#!/usr/bin/env python3
"""CLI for InjecAgent evaluation"""

import argparse
import json
import torch
from .injecagent import InjecAgentEvaluator


def main():
    parser = argparse.ArgumentParser(description="Run InjecAgent evaluation")
    parser.add_argument("--model_path", type=str, required=True, help="HF model id or local adapter dir")
    parser.add_argument("--setting", choices=["base", "enhanced"], default="base")
    parser.add_argument("--prompt_type", choices=["InjecAgent", "hwchase17_react"], default="InjecAgent")
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--num_samples", type=int, default=None, help="Limit number of test cases")
    parser.add_argument("--only_first_step", action="store_true", help="Skip DS stage-2 (S2)")
    parser.add_argument("--device_map", type=str, default="auto")
    parser.add_argument("--dtype", type=str, default="float16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--out_dir", type=str, default="outputs/injecagent", help="Output directory root")
    parser.add_argument("--batch_size", type=int, default=8,
                       help="Batch size for GPU-accelerated generation (default: 8)")
    parser.add_argument(
        "--record_path",
        type=str,
        default=None,
        help="Optional txt path to append a one-line JSON record of this eval run (default: outputs/eval_records.txt)",
    )
    
    args = parser.parse_args()
    
    dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
    evaluator = InjecAgentEvaluator(
        args.model_path,
        max_new_tokens=args.max_new_tokens,
        device_map=args.device_map,
        torch_dtype=dtype_map[args.dtype],
    )
    
    results = evaluator.evaluate(
        setting=args.setting,
        prompt_type=args.prompt_type,
        num_samples=args.num_samples,
        only_first_step=args.only_first_step,
        out_dir=args.out_dir,
        batch_size=args.batch_size,
        record_path=args.record_path,
    )
    
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()

