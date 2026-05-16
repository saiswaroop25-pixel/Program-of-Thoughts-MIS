"""
extensions/adaptive_runner.py
──────────────────────────────
🆕 NEW — Ties all extensions together into one adaptive pipeline.

Combines:
  1. ComplexityAnalyzer  → pick the right method per question
  2. PoT / CoT / Hybrid  → generate and execute
  3. MathVerifier        → symbolically double-check answers
  4. Self-Consistency    → for low-confidence answers, sample K times
  5. ErrorClassifier     → categorise failures post-hoc

This is the "upgraded" PoT system for your maths project —
it goes beyond the paper by being method-adaptive and self-verifying.

Usage:
    python -m extensions.adaptive_runner --start 0 --end 50 --model gpt-4o-mini
"""

import argparse
import json
import os
import time
from tqdm import tqdm

import config
from utils.data_loader        import load_gsm8k
from utils.executor           import execute_program
from utils.evaluator          import compute_accuracy
from utils.llm_client         import get_completion
from prompts.gsm8k_few_shot   import build_few_shot_prompt
from prompts.zero_shot        import build_zero_shot_pot
from extensions.complexity_analyzer import route_method
from extensions.math_verifier       import verify_answer
from extensions.self_consistency    import self_consistent_pot
from extensions.pot_cot_hybrid      import hybrid_pot_cot
from extensions.error_classifier    import classify_errors, summarise_errors


def run_adaptive(args):
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    print(f"\n{'='*55}")
    print(f"  Adaptive PoT Runner")
    print(f"  Model  : {args.model}")
    print(f"  Dataset: GSM8K [{args.start}:{args.end}]")
    print(f"  SC     : {'enabled (K=' + str(args.sc_k) + ')' if args.use_sc else 'disabled'}")
    print(f"  Verify : {'enabled' if args.verify else 'disabled'}")
    print(f"{'='*55}\n")

    dataset = load_gsm8k(split="test", start=args.start, end=args.end)
    print(f"Loaded {len(dataset)} examples.\n")

    results = []
    method_counts = {"pot_only": 0, "pot_verify": 0, "pot_cot_hybrid": 0, "cot_only": 0}

    for i, item in enumerate(tqdm(dataset, desc="Adaptive PoT")):
        question = item["question"]

        # ── Step 1: Route to best method ───────────────────────────────────
        method, analysis = route_method(question)
        method_counts[method] = method_counts.get(method, 0) + 1

        result_item = {
            "index":            args.start + i,
            "question":         question,
            "answer":           item["answer"],
            "method":           method,
            "category":         analysis.category,
            "complexity_score": analysis.complexity_score,
            "confidence":       analysis.confidence,
        }

        # ── Step 2: Generate answer ─────────────────────────────────────────
        if method == "pot_cot_hybrid":
            hybrid = hybrid_pot_cot(question, model=args.model)
            code      = hybrid["pot_code"]
            answer    = hybrid["answer"]
            exec_err  = hybrid["exec_error"]

        elif method == "cot_only":
            system = "You are an expert mathematician. Solve step by step. End with 'The answer is: X'."
            sys_p, user_p = (system, f"Question: {question}\n\nLet's think step by step:")
            completions = get_completion(user_p, sys_p, model=args.model, temperature=0.0, n=1)
            from prompts.zero_shot import extract_cot_answer
            code    = completions[0] if completions else ""
            answer  = extract_cot_answer(code)
            exec_err = None

        else:
            # pot_only or pot_verify → generate code and execute
            if args.use_sc and analysis.confidence < 0.8:
                # Low confidence → use self-consistency
                sc_result = self_consistent_pot(
                    question=question,
                    build_prompt_fn=lambda q: build_few_shot_prompt(q, n_shots=4),
                    model=args.model,
                    k=args.sc_k,
                    temperature=0.4,
                )
                code     = sc_result["codes"][0] if sc_result["codes"] else ""
                answer   = sc_result["answer"]
                exec_err = None
                result_item["sc_confidence"] = sc_result["confidence"]
            else:
                # High confidence or SC disabled → greedy
                prompt = build_few_shot_prompt(question, n_shots=4)
                system = "Write Python code to solve. Store answer in `ans`. No markdown."
                completions = get_completion(prompt, system, model=args.model, temperature=0.0, n=1)
                code    = completions[0] if completions else ""
                answer, exec_err = execute_program(code, timeout=config.CODE_TIMEOUT_SECS)

        result_item.update({"code": code, "prediction": answer, "exec_error": exec_err})

        # ── Step 3: Symbolic verification ───────────────────────────────────
        if args.verify and method in ("pot_only", "pot_verify") and answer is not None:
            verification = verify_answer(code=code, predicted=answer, question=question)
            result_item["verified"]            = verification["verified"]
            result_item["sympy_answer"]        = verification["sympy_answer"]
            result_item["verification_method"] = verification["method"]
            result_item["verification_notes"]  = verification["notes"]

            # If SymPy refutes the answer and we haven't used SC, try once more
            if verification["verified"] is False and not args.use_sc:
                tqdm.write(f"  [Verify] Q{args.start+i}: SymPy refuted {answer}. Re-sampling...")
                sc_result = self_consistent_pot(
                    question=question,
                    build_prompt_fn=lambda q: build_few_shot_prompt(q, n_shots=6),
                    model=args.model,
                    k=5,
                    temperature=0.4,
                )
                if sc_result["answer"] is not None:
                    result_item["prediction"]          = sc_result["answer"]
                    result_item["verified"]            = None
                    result_item["verification_notes"] += " [re-sampled after refutation]"

        results.append(result_item)
        time.sleep(0.05)

    # ── Scoring ──────────────────────────────────────────────────────────────
    metrics = compute_accuracy(results, mode="exact")

    print(f"\n{'='*55}")
    print(f"  ADAPTIVE PoT RESULTS")
    print(f"  Correct : {metrics['correct']} / {metrics['total']}")
    print(f"  Accuracy: {metrics['accuracy']:.1%}")
    print(f"\n  Method distribution:")
    for m, c in method_counts.items():
        print(f"    {m:20s}: {c}")

    if args.verify:
        verified_count = sum(1 for r in results if r.get("verified") is True)
        refuted_count  = sum(1 for r in results if r.get("verified") is False)
        print(f"\n  Verification: {verified_count} confirmed, {refuted_count} refuted by SymPy")
    print(f"{'='*55}\n")

    # ── Error classification ─────────────────────────────────────────────────
    if args.classify_errors and metrics["errors"]:
        print(f"Classifying {len(metrics['errors'])} errors...")
        classified = classify_errors(metrics["errors"][:20], model=args.model)  # limit to 20
        print(summarise_errors(classified))

    # ── Save results ─────────────────────────────────────────────────────────
    tag     = f"adaptive_{args.model}_{args.start}_{args.end}"
    outfile = os.path.join(config.OUTPUT_DIR, f"gsm8k_{tag}.jsonl")
    with open(outfile, "w") as f:
        for r in results:
            f.write(json.dumps(r, default=str) + "\n")

    print(f"Results saved to: {outfile}")
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Adaptive PoT Runner")
    parser.add_argument("--model",           default=config.DEFAULT_MODEL)
    parser.add_argument("--start",           type=int,  default=0)
    parser.add_argument("--end",             type=int,  default=50)
    parser.add_argument("--use_sc",          action="store_true", help="Enable self-consistency")
    parser.add_argument("--sc_k",            type=int,  default=5, help="SC samples")
    parser.add_argument("--verify",          action="store_true", help="Enable SymPy verification")
    parser.add_argument("--classify_errors", action="store_true", help="Classify failures with LLM")
    args = parser.parse_args()
    run_adaptive(args)
