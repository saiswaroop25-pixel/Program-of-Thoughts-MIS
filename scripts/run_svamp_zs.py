"""
scripts/run_svamp_zs.py
────────────────────────
Zero-shot PoT on SVAMP (Simple Variation on Arithmetic Math word Problems).

SVAMP tests robustness — questions are slight variations of arithmetic problems
designed to expose models that rely on superficial patterns.

Paper result (Codex PoT zero-shot): 70.8%
Paper result (GPT-3 CoT zero-shot): 63.7%

Usage:
    python scripts/run_svamp_zs.py --start 0 --end 100
    python scripts/run_svamp_zs.py --start 0 --end 1000   # full test set
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
import time
from tqdm import tqdm

import config
from utils.data_loader  import load_svamp
from utils.executor     import execute_program
from utils.evaluator    import compute_accuracy
from utils.llm_client   import get_completion
from prompts.zero_shot  import build_zero_shot_pot


def _unwrap_solver(code: str) -> str:
    """If LLM returned a solver() function, append a call so `ans` gets set."""
    if "def solver()" in code and "ans = solver()" not in code:
        return code + "\nans = solver()"
    return code


def run(args):
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    print(f"\nZero-shot PoT | SVAMP | {config.DEFAULT_PROVIDER} / {config.DEFAULT_MODEL}")
    print(f"Range: {args.start} to {args.end}")
    print(f"Paper Codex baseline: 70.8%  |  Paper GPT-3 CoT: 63.7%\n")

    dataset = load_svamp(start=args.start, end=args.end)
    print(f"Loaded {len(dataset)} examples.\n")

    tag     = f"svamp_zs_{config.DEFAULT_MODEL.replace('/', '-')}_{args.start}_{args.end}"
    outfile = os.path.join(config.OUTPUT_DIR, f"{tag}.jsonl")

    # Resume support — skip already-done indices
    done_indices = set()
    existing     = []
    if os.path.exists(outfile):
        with open(outfile, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    r = json.loads(line)
                    done_indices.add(r["index"])
                    existing.append(r)
        if done_indices:
            print(f"Resuming — {len(done_indices)} questions already done.\n")

    results = list(existing)

    for i, item in enumerate(tqdm(dataset, desc="PoT SVAMP zero-shot")):
        idx = args.start + i
        if idx in done_indices:
            continue

        system_prompt, user_prompt = build_zero_shot_pot(item["question"])

        completions = get_completion(
            prompt=user_prompt,
            system_prompt=system_prompt,
            temperature=0.0,
            n=1,
        )
        code   = _unwrap_solver(completions[0] if completions else "")
        answer, exec_error = execute_program(code, timeout=config.CODE_TIMEOUT_SECS)

        result = {
            "index":      idx,
            "question":   item["question"],
            "answer":     item["answer"],
            "equation":   item.get("equation", ""),
            "code":       code,
            "prediction": answer,
            "exec_error": exec_error,
        }
        results.append(result)

        # Write immediately so you can resume after a Groq rate limit
        with open(outfile, "a", encoding="utf-8") as f:
            f.write(json.dumps(result, default=str) + "\n")

        time.sleep(0.15)   # stay within Groq's per-minute rate limit

    # ── Score ─────────────────────────────────────────────────────────────────
    metrics = compute_accuracy(results, mode="exact")

    exec_errors = sum(1 for r in results if r.get("exec_error"))
    missing     = sum(1 for r in results if r.get("prediction") is None)

    print(f"\n{'='*52}")
    print(f"  Zero-shot PoT | SVAMP")
    print(f"  Model   : {config.DEFAULT_MODEL}")
    print(f"  Correct : {metrics['correct']} / {metrics['total']}")
    print(f"  Accuracy: {metrics['accuracy']:.1%}")
    print(f"  Exec errors   : {exec_errors} ({exec_errors/metrics['total']:.0%})")
    print(f"  Missing pred. : {missing} ({missing/metrics['total']:.0%})")
    print(f"  Paper Codex PoT  ZS: 70.8%")
    print(f"  Paper GPT-3 CoT  ZS: 63.7%")
    print(f"{'='*52}\n")
    print(f"Results saved to: {outfile}")

    # Also save a small error file for analysis
    errfile = os.path.join(config.OUTPUT_DIR, f"{tag}_errors.json")
    with open(errfile, "w", encoding="utf-8") as f:
        json.dump(metrics["errors"][:50], f, indent=2, default=str)
    print(f"Top-50 errors  : {errfile}")

    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Zero-shot PoT on SVAMP")
    parser.add_argument("--start", type=int, default=0,
                        help="Start index (inclusive)")
    parser.add_argument("--end",   type=int, default=100,
                        help="End index (exclusive). Full set is 1000.")
    run(parser.parse_args())
