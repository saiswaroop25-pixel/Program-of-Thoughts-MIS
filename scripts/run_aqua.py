"""
scripts/run_aqua.py
────────────────────
Few-shot PoT on AQuA-RAT (algebraic multiple-choice).
Uses SymPy for polynomial/algebraic solving.

Usage:
    python scripts/run_aqua.py --start 0 --end 100
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
import time
from tqdm import tqdm

import config
from utils.data_loader      import load_aqua
from utils.executor         import execute_program
from utils.evaluator        import compute_accuracy, option_match
from utils.llm_client       import get_completion
from prompts.aqua_few_shot  import build_aqua_prompt


def _prepare_aqua_code(raw_code: str) -> str:
    """Keep only the final-question solution when the model echoes exemplars."""
    code = raw_code.strip()
    if "```" in code:
        code = code.replace("```python", "```").strip()
        parts = code.split("```")
        code = max(parts, key=len).strip() if parts else code

    marker = "# Python code for this question only:"
    if marker in code:
        code = code.split(marker)[-1].strip()
    elif code.count("# Question:") > 1:
        code = "# Question:" + code.split("# Question:")[-1]
        lines = code.splitlines()
        # Drop the final question/comment line and keep the actual program.
        code = "\n".join(line for line in lines[1:] if not line.strip().startswith("# Answer options:")).strip()
    elif code.count("# Question:") == 1:
        lines = code.splitlines()
        code = "\n".join(line for line in lines[1:] if not line.strip().startswith("# Answer options:")).strip()

    prefix = "\n".join([
        "import math",
        "from sympy import Symbol, symbols, simplify, solve, sqrt, Rational, Eq",
        "",
    ])
    return prefix + code


def run(args):
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    print(f"\nFew-shot PoT | AQuA | {config.DEFAULT_PROVIDER} / {config.DEFAULT_MODEL}")
    print(f"Examples: {args.start} to {args.end}\n")

    dataset = load_aqua(split="test", start=args.start, end=args.end)
    print(f"Loaded {len(dataset)} examples.")

    system = (
        "You are a Python programmer solving algebraic word problems. "
        "Use sympy for equations. Always store the final answer in `ans`. "
        "Write only raw Python code."
    )

    results = []
    for i, item in enumerate(tqdm(dataset, desc="PoT AQuA")):
        prompt = build_aqua_prompt(item["question"], item["options"], n_shots=8)
        completions = get_completion(prompt=prompt, system_prompt=system, temperature=0.0, n=1)
        raw_code = completions[0] if completions else ""
        code = _prepare_aqua_code(raw_code)

        numeric_answer, exec_error = execute_program(code, timeout=config.CODE_TIMEOUT_SECS)

        # Map numeric answer to closest option letter (A-E)
        predicted_option = option_match(numeric_answer, item["options"])

        results.append({
            "index":            args.start + i,
            "question":         item["question"],
            "options":          item["options"],
            "answer":           item["answer"],        # letter e.g. "B"
            "raw_code":         raw_code,
            "code":             code,
            "numeric_answer":   numeric_answer,
            "prediction":       predicted_option,
            "exec_error":       exec_error,
        })
        time.sleep(0.15)

    metrics = compute_accuracy(results, mode="option")
    print(f"\n{'='*50}")
    print(f"  PoT Few-shot | AQuA")
    print(f"  Correct : {metrics['correct']} / {metrics['total']}")
    print(f"  Accuracy: {metrics['accuracy']:.1%}  (paper Codex baseline: 54.1%)")
    print(f"{'='*50}\n")

    tag = f"aqua_fs_{config.DEFAULT_MODEL.replace('/', '-')}_{args.start}_{args.end}"
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
