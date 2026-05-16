"""
compute_score.py
─────────────────
Score a JSONL output file from any of the run_*.py scripts.

Usage:
    python compute_score.py --inputs outputs/gsm8k_zs_gpt-4o-mini_0_100.jsonl
    python compute_score.py --inputs outputs/gsm8k_fs_gpt-4o-mini_0_100.jsonl --relaxed
    python compute_score.py --inputs outputs/aqua_fs_gpt-4o-mini_0_100.jsonl --mode option
"""

import argparse
import json
import os
from utils.evaluator import compute_accuracy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs",  required=True, help="Path to .jsonl results file")
    parser.add_argument("--mode",    default="exact", choices=["exact", "relaxed", "option"])
    parser.add_argument("--relaxed", action="store_true", help="Shorthand for --mode relaxed")
    args = parser.parse_args()

    if args.relaxed:
        args.mode = "relaxed"

    if not os.path.exists(args.inputs):
        print(f"File not found: {args.inputs}")
        return

    results = []
    with open(args.inputs) as f:
        for line in f:
            line = line.strip()
            if line:
                results.append(json.loads(line))

    if not results:
        print("No results found in file.")
        return

    metrics = compute_accuracy(results, mode=args.mode)

    print(f"\n{'='*50}")
    print(f"  File    : {args.inputs}")
    print(f"  Mode    : {args.mode}")
    print(f"  Correct : {metrics['correct']} / {metrics['total']}")
    print(f"  Accuracy: {metrics['accuracy']:.1%}  ({metrics['accuracy']*100:.2f}%)")
    print(f"{'='*50}")

    # Execution error rate
    errors_with_exec = [r for r in results if r.get("exec_error")]
    if errors_with_exec:
        rate = len(errors_with_exec) / len(results)
        print(f"\n  Execution error rate: {rate:.1%} ({len(errors_with_exec)}/{len(results)})")
        err_types: dict = {}
        for r in errors_with_exec:
            err = str(r.get("exec_error", ""))[:50]
            err_type = err.split(":")[0].strip()
            err_types[err_type] = err_types.get(err_type, 0) + 1
        print("  Error breakdown:")
        for t, c in sorted(err_types.items(), key=lambda x: -x[1])[:5]:
            print(f"    {t}: {c}")


if __name__ == "__main__":
    main()
