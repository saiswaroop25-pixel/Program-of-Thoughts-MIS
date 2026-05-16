"""
scripts/run_cot_baseline.py
────────────────────────────
CoT baseline — runs on same questions as PoT for direct comparison.

Usage:
    python scripts/run_cot_baseline.py --start 0 --end 100
    python scripts/run_cot_baseline.py --start 0 --end 100 --zero_shot
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
import time
from tqdm import tqdm

import config
from utils.data_loader  import load_gsm8k
from utils.evaluator    import compute_accuracy
from utils.llm_client   import get_completion
from prompts.zero_shot  import build_zero_shot_cot, extract_cot_answer

COT_EXEMPLARS = [
    ("Janet's ducks lay 16 eggs per day. She eats 3 and bakes 4. She sells the rest for $2 each. How much per day?",
     "She uses 3+4=7 eggs. Sells 16-7=9 eggs. Earns 9×2=$18. The answer is: 18"),
    ("A robe takes 2 bolts blue fiber and half that white fiber. How many bolts total?",
     "White fiber = 2/2 = 1 bolt. Total = 2+1 = 3. The answer is: 3"),
    ("Josh buys a house for $80,000 and puts $50,000 in repairs. Value increased 150%. Profit?",
     "Increase = 80000×1.5 = $120,000. New value = $200,000. Profit = 200000-80000-50000 = $70,000. The answer is: 70000"),
]

def run(args):
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    mode = "zero-shot" if args.zero_shot else "few-shot"
    print(f"\nCoT {mode} | GSM8K | {config.DEFAULT_PROVIDER} / {config.DEFAULT_MODEL}")
    print(f"Examples: {args.start} to {args.end}\n")

    dataset = load_gsm8k(split="test", start=args.start, end=args.end)

    system = "Solve math problems step by step. End with 'The answer is: X' where X is numeric."
    results = []

    for i, item in enumerate(tqdm(dataset, desc=f"CoT {mode}")):
        if args.zero_shot:
            _, user_prompt = build_zero_shot_cot(item["question"])
        else:
            ex_text = "\n\n".join(f"Q: {q}\nA: {a}" for q, a in COT_EXEMPLARS)
            user_prompt = ex_text + f"\n\nQ: {item['question']}\nA:"

        completions = get_completion(
            prompt=user_prompt, system_prompt=system, temperature=0.0, n=1
        )
        response = completions[0] if completions else ""
        answer   = extract_cot_answer(response)

        results.append({
            "index":      args.start + i,
            "question":   item["question"],
            "answer":     item["answer"],
            "response":   response,
            "prediction": answer,
        })
        time.sleep(0.1)

    metrics = compute_accuracy(results, mode="exact")
    paper_ref = "40.5%" if args.zero_shot else "63.1%"
    print(f"\n{'='*50}")
    print(f"  CoT {mode} | GSM8K")
    print(f"  Correct : {metrics['correct']} / {metrics['total']}")
    print(f"  Accuracy: {metrics['accuracy']:.1%}  (paper Codex CoT: {paper_ref})")
    print(f"{'='*50}\n")

    tag = f"gsm8k_cot_{'zs' if args.zero_shot else 'fs'}_{config.DEFAULT_MODEL.replace('/', '-')}_{args.start}_{args.end}"
    outfile = os.path.join(config.OUTPUT_DIR, f"{tag}.jsonl")
    with open(outfile, "w") as f:
        for r in results:
            f.write(json.dumps(r, default=str) + "\n")
    print(f"Saved: {outfile}")
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start",     type=int, default=0)
    parser.add_argument("--end",       type=int, default=100)
    parser.add_argument("--zero_shot", action="store_true")
    run(parser.parse_args())
