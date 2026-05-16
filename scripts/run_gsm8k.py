"""
scripts/run_gsm8k.py
─────────────────────
Few-shot PoT on GSM8K. Replicates Table 2 of the paper.
Paper result: 71.6% (Codex). Modern models typically exceed this.

Usage:
    python scripts/run_gsm8k.py --start 0 --end 100
    python scripts/run_gsm8k.py --start 0 --end 100 --shots 4
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
import time
from tqdm import tqdm

import config
from utils.data_loader       import load_gsm8k
from utils.executor          import execute_program
from utils.evaluator         import compute_accuracy
from utils.llm_client        import get_completion
from prompts.gsm8k_few_shot  import build_few_shot_prompt


def run(args):
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    print(f"\nFew-shot PoT | GSM8K | {config.DEFAULT_PROVIDER} / {config.DEFAULT_MODEL}")
    print(f"Examples: {args.start} to {args.end}\n")

    dataset = load_gsm8k(split="test", start=args.start, end=args.end)
    print(f"Loaded {len(dataset)} examples.")

    results = []
    system = (
        "You are a Python programmer solving math word problems. "
        "Write only executable Python code — no markdown, no comments explaining reasoning. "
        "Always store the final numeric answer in a variable called `ans`."
    )

    for i, item in enumerate(tqdm(dataset, desc="PoT few-shot")):
        prompt = build_few_shot_prompt(item["question"], n_shots=args.shots)
        completions = get_completion(prompt=prompt, system_prompt=system, temperature=0.0, n=1)
        code = completions[0] if completions else ""
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
    _print_results("PoT Few-shot", metrics, paper_baseline=71.6)

    tag = f"gsm8k_fs_{config.DEFAULT_MODEL.replace('/', '-')}_{args.start}_{args.end}"
    _save(results, metrics, tag)
    return metrics


def _print_results(name, metrics, paper_baseline):
    print(f"\n{'='*50}")
    print(f"  {name} | GSM8K")
    print(f"  Correct : {metrics['correct']} / {metrics['total']}")
    print(f"  Accuracy: {metrics['accuracy']:.1%}  (paper baseline: {paper_baseline}%)")
    print(f"{'='*50}\n")


def _save(results, metrics, tag):
    outfile = os.path.join(config.OUTPUT_DIR, f"{tag}.jsonl")
    with open(outfile, "w") as f:
        for r in results:
            f.write(json.dumps(r, default=str) + "\n")
    errfile = os.path.join(config.OUTPUT_DIR, f"{tag}_errors.json")
    with open(errfile, "w") as f:
        json.dump(metrics["errors"], f, indent=2)
    print(f"Saved: {outfile}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end",   type=int, default=100)
    parser.add_argument("--shots", type=int, default=8)
    run(parser.parse_args())
