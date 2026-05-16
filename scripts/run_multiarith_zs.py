"""Zero-shot PoT on MultiArith.

This mirrors the original PoT repo's zero-shot MultiArith experiment while
using the shared provider/client/executor in this project.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from prompts.zero_shot import build_zero_shot_pot
from utils.data_loader import load_multiarith
from utils.evaluator import compute_accuracy
from utils.executor import execute_program
from utils.llm_client import get_completion


def _unwrap_solver(code: str) -> str:
    if "def solver()" in code and "ans = solver()" not in code:
        return code + "\nans = solver()"
    return code


def run(args):
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    print(f"\nZero-shot PoT | MultiArith | {config.DEFAULT_PROVIDER} / {config.DEFAULT_MODEL}")
    print(f"Examples: {args.start} to {args.end}\n")

    dataset = load_multiarith(start=args.start, end=args.end)
    print(f"Loaded {len(dataset)} examples.")

    results = []
    for i, item in enumerate(tqdm(dataset, desc="PoT MultiArith zero-shot")):
        system_prompt, user_prompt = build_zero_shot_pot(item["question"])
        completions = get_completion(
            prompt=user_prompt,
            system_prompt=system_prompt,
            temperature=0.0,
            n=1,
        )
        code = _unwrap_solver(completions[0] if completions else "")
        answer, exec_error = execute_program(code, timeout=config.CODE_TIMEOUT_SECS)

        results.append({
            "index": args.start + i,
            "question": item["question"],
            "answer": item["answer"],
            "code": code,
            "prediction": answer,
            "exec_error": exec_error,
        })
        time.sleep(0.1)

    metrics = compute_accuracy(results, mode="exact")
    print(f"\n{'=' * 50}")
    print("  Zero-shot PoT | MultiArith")
    print(f"  Correct : {metrics['correct']} / {metrics['total']}")
    print(f"  Accuracy: {metrics['accuracy']:.1%}  (paper zero-shot PoT: 92.2%)")
    print(f"{'=' * 50}\n")

    tag = f"multiarith_zs_{config.DEFAULT_MODEL.replace('/', '-')}_{args.start}_{args.end}"
    outfile = os.path.join(config.OUTPUT_DIR, f"{tag}.jsonl")
    with open(outfile, "w", encoding="utf-8") as handle:
        for row in results:
            handle.write(json.dumps(row, default=str) + "\n")
    print(f"Saved: {outfile}")
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=100)
    run(parser.parse_args())
