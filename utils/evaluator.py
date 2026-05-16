"""
utils/evaluator.py
──────────────────
Scoring functions matching the paper's evaluation protocol:
  • Exact match (GSM8K, SVAMP, MultiArith)
  • Relaxed match with math.isclose (FinQA — large float numbers)
  • Option matching (AQuA — multiple choice)
"""

import math
import re
from typing import Any, Optional, Union


# ── Exact Match ───────────────────────────────────────────────────────────────
def exact_match(prediction: Any, reference: Union[str, float, int],
                round_digits: int = 4) -> bool:
    """
    Round both values to `round_digits` and compare.
    Handles string references like '18.5' and numeric predictions.
    """
    pred = _to_float(prediction)
    ref  = _to_float(reference)
    if pred is None or ref is None:
        return str(prediction).strip() == str(reference).strip()
    return round(pred, round_digits) == round(ref, round_digits)


def relaxed_match(prediction: Any, reference: Any,
                  rel_tol: float = 1e-3) -> bool:
    """
    math.isclose with relative tolerance — used for FinQA (large numbers).
    """
    pred = _to_float(prediction)
    ref  = _to_float(reference)
    if pred is None or ref is None:
        return str(prediction).strip() == str(reference).strip()
    return math.isclose(pred, ref, rel_tol=rel_tol)


# ── AQuA Option Matching ──────────────────────────────────────────────────────
def option_match(numeric_answer: Any, options: list[str]) -> Optional[str]:
    """
    Given a computed numeric answer, find the closest option letter (A-E).
    Used for AQuA where the final answer must be a labelled option.

    Returns the letter ('A', 'B', ...) or None if no options can be parsed.
    """
    numeric = _to_float(numeric_answer)
    if numeric is None or not options:
        return None

    best_letter = None
    best_diff   = float("inf")

    for opt in options:
        # options look like ['A)18', 'B)24', ...]
        match = re.match(r"([A-E])\s*[):]?\s*(.+)", opt.strip())
        if not match:
            continue
        letter = match.group(1)
        val    = _to_float(match.group(2))
        if val is None:
            continue
        diff = abs(numeric - val)
        if diff < best_diff:
            best_diff   = diff
            best_letter = letter

    return best_letter


# ── Batch Scoring ─────────────────────────────────────────────────────────────
def compute_accuracy(results: list[dict],
                     mode: str = "exact",
                     answer_key: str = "answer",
                     pred_key: str = "prediction") -> dict:
    """
    Compute accuracy over a list of result dicts.

    Parameters
    ----------
    results    : List of dicts, each with at least `answer_key` and `pred_key`.
    mode       : "exact" | "relaxed" | "option"
    answer_key : Key for the ground truth answer.
    pred_key   : Key for the model's prediction.

    Returns
    -------
    dict with keys: correct, total, accuracy, errors
    """
    correct = 0
    errors  = []

    for i, item in enumerate(results):
        pred = item.get(pred_key)
        ref  = item.get(answer_key)

        if mode == "exact":
            is_correct = exact_match(pred, ref)
        elif mode == "relaxed":
            is_correct = relaxed_match(pred, ref)
        elif mode == "option":
            options = item.get("options", [])
            if isinstance(pred, str) and pred.strip().upper() in {"A", "B", "C", "D", "E"}:
                chosen = pred.strip().upper()
            else:
                chosen = option_match(pred, options)
            is_correct = (chosen == ref)
        else:
            raise ValueError(f"Unknown mode: {mode}")

        if is_correct:
            correct += 1
        else:
            errors.append({
                "index":      i,
                "question":   item.get("question", ""),
                "prediction": pred,
                "reference":  ref,
                "code":       item.get("code", ""),
                "exec_error": item.get("exec_error", ""),
            })

    total = len(results)
    return {
        "correct":  correct,
        "total":    total,
        "accuracy": correct / total if total > 0 else 0.0,
        "errors":   errors,
    }


# ── Helper ────────────────────────────────────────────────────────────────────
def _to_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, (list, tuple)):
        flattened = _flatten_numeric(value)
        return flattened[0] if len(flattened) == 1 else None
    s = str(value).strip().replace(",", "")
    s = (
        s.replace("$", "")
        .replace("Rs.", "")
        .replace("Rs", "")
        .replace("√", "sqrt")
        .replace("–", "-")
        .replace("−", "-")
    )
    s = re.sub(r"\s+", "", s)
    s = re.sub(r"sqrt(\d+(?:\.\d+)?)", r"sqrt(\1)", s)
    s = re.sub(r"(?<=\d)(?=sqrt|\()", "*", s)
    s = re.sub(r"(?<=\))(?=\d|sqrt|\()", "*", s)
    if s.endswith("%"):
        s = s[:-1]
    if re.fullmatch(r"[-+]?\d+(?:\.\d+)?[A-Za-z]+", s):
        s = re.match(r"[-+]?\d+(?:\.\d+)?", s).group(0)
    # Handle fractions like "3/4"
    if "/" in s:
        try:
            num, den = s.split("/")
            return float(num) / float(den)
        except Exception:
            pass
    try:
        return float(s)
    except (ValueError, TypeError):
        pass
    try:
        import sympy
        val = sympy.sympify(s, locals={"sqrt": sympy.sqrt})
        if getattr(val, "is_real", False):
            return float(val.evalf())
    except Exception:
        return None
    return None


def _flatten_numeric(value: Any) -> list[float]:
    if isinstance(value, (list, tuple)):
        out = []
        for item in value:
            out.extend(_flatten_numeric(item))
        return out
    num = _to_float(value)
    return [] if num is None else [num]
