#!/usr/bin/env python3
"""
Generate a universal GCG-style adversarial suffix (offline) and print/save it.

This integrates with existing ASR eval by passing the output as `--injection`.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import List

import torch

from attacks.gcg import GCGConfig, generate_universal_gcg_suffix
from utils import load_data, load_model_auto


def _load_prompts(dataset_path: str | None, num_samples: int | None) -> List[str]:
    if dataset_path is None:
        # Match eval/asr.py behavior: try AlpacaEval, fallback to local test dataset.
        try:
            import alpaca_eval

            data = alpaca_eval.get_data_alpaca_eval()
        except Exception:
            data = load_data("datasets/alpaca_data_with_input_test.jsonl")
    else:
        data = load_data(dataset_path)

    if num_samples is not None:
        data = data[: min(num_samples, len(data))]

    prompts: List[str] = []
    for item in data:
        instruction = item.get("instruction", "") or ""
        input_text = item.get("input", "") or ""
        if input_text:
            prompts.append(f"{instruction}\n\n{input_text}")
        else:
            prompts.append(instruction)
    prompts = [p for p in prompts if p.strip() != ""]
    return prompts


def main():
    p = argparse.ArgumentParser(description="Generate a universal GCG adversarial suffix")
    p.add_argument("--model_path", type=str, required=True, help="HF model id or local path (or adapter dir)")
    p.add_argument("--dataset_path", type=str, default=None, help="Optional json/jsonl with instruction/input fields")
    p.add_argument("--num_samples", type=int, default=64, help="How many prompts to optimize over")
    p.add_argument("--target", type=str, default="Hacked", help="Target string to increase probability of")

    # GCG config
    p.add_argument("--suffix_len", type=int, default=20)
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--max_length", type=int, default=512)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--allow_non_ascii", action="store_true", help="Allow non-ascii tokens in suffix search")
    p.add_argument("--init_suffix", type=str, default=None, help="Optional init string for suffix")

    p.add_argument("--output_path", type=str, default=None, help="Optional path to save JSON with suffix/metadata")

    args = p.parse_args()

    print(f"[GCG] Loading model: {args.model_path}")
    model, tokenizer = load_model_auto(
        args.model_path,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )

    prompts = _load_prompts(args.dataset_path, args.num_samples)
    if len(prompts) == 0:
        raise SystemExit("No prompts loaded (empty dataset?)")
    print(f"[GCG] Loaded {len(prompts)} prompts")

    cfg = GCGConfig(
        suffix_len=args.suffix_len,
        steps=args.steps,
        seed=args.seed,
        batch_size=args.batch_size,
        max_length=args.max_length,
        allow_non_ascii=args.allow_non_ascii,
        device=None,
    )

    suffix = generate_universal_gcg_suffix(
        model=model,
        tokenizer=tokenizer,
        prompts=prompts,
        target=args.target,
        model_path=args.model_path,
        cfg=cfg,
        init_suffix=args.init_suffix,
    )

    print("\n=== UNIVERSAL GCG SUFFIX ===")
    print(suffix)

    if args.output_path is not None:
        os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)
        payload = {
            "model_path": args.model_path,
            "dataset_path": args.dataset_path,
            "num_samples": len(prompts),
            "target": args.target,
            "suffix": suffix,
            "cfg": {
                "suffix_len": cfg.suffix_len,
                "steps": cfg.steps,
                "seed": cfg.seed,
                "batch_size": cfg.batch_size,
                "max_length": cfg.max_length,
                "allow_non_ascii": cfg.allow_non_ascii,
            },
        }
        with open(args.output_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        print(f"[GCG] Saved to {args.output_path}")


if __name__ == "__main__":
    main()


