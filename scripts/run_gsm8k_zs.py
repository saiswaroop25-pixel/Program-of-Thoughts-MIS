"""
scripts/run_gsm8k_zs.py
────────────────────────
Zero-shot PoT on GSM8K. Replicates Table 3 of the paper.
Paper result: 57.0% (GPT-3). Free models (Gemini/Groq) exceed this.

Usage:
    python scripts/run_gsm8k_zs.py --start 0 --end 100
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
import time
from tqdm import tqdm

import config
from utils.data_loader   import load_gsm8k
from utils.executor      import execute_program
from utils.evaluator     import compute_accuracy
from utils.llm_client    import get_completion
from prompts.zero_shot   import build_zero_shot_pot


def _unwrap_solver(code: str) -> str:
    """If LLM returned a solver() function, call it so `ans` gets set."""
    if "def solver()" in code and "ans = solver()" not in code:
        return code + "\nans = solver()"
    return code


def run(args):
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    print(f"\nZero-shot PoT | GSM8K | {config.DEFAULT_PROVIDER} / {config.DEFAULT_MODEL}")
    print(f"Examples: {args.start} to {args.end}\n")

    dataset = load_gsm8k(split="test", start=args.start, end=args.end)
    print(f"Loaded {len(dataset)} examples.")

    results = []
    for i, item in enumerate(tqdm(dataset, desc="PoT zero-shot")):
        system_prompt, user_prompt = build_zero_shot_pot(item["question"])
        completions = get_completion(
            prompt=user_prompt, system_prompt=system_prompt, temperature=0.0, n=1
        )
        code   = _unwrap_solver(completions[0] if completions else "")
        answer, exec_error = execute_program(code, timeout=config.CODE_TIMEOUT_SECS)

        results.append({
            "index":      args.start + i,
            "question":   item["question"],
            "answer":     item["answer"],
            "code":       code,
            "prediction": answer,
            "exec_error": exec_error,
        })
        time.sleep(0.1)

    metrics = compute_accuracy(results, mode="exact")
    print(f"\n{'='*50}")
    print(f"  Zero-shot PoT | GSM8K")
    print(f"  Correct : {metrics['correct']} / {metrics['total']}")
    print(f"  Accuracy: {metrics['accuracy']:.1%}  (paper GPT-3 baseline: 57.0%)")
    print(f"{'='*50}\n")

    tag = f"gsm8k_zs_{config.DEFAULT_MODEL.replace('/', '-')}_{args.start}_{args.end}"
    outfile = os.path.join(config.OUTPUT_DIR, f"{tag}.jsonl")
    with open(outfile, "w") as f:
        for r in results:
            f.write(json.dumps(r, default=str) + "\n")
    print(f"Saved: {outfile}")
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end",   type=int, default=100)
    run(parser.parse_args())
