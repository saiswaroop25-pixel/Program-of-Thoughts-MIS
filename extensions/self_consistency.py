"""
extensions/self_consistency.py
───────────────────────────────
Self-Consistency (SC) decoding — Section 3.2 of the paper.

Instead of greedy decoding (temperature=0), SC:
  1. Samples K completions at temperature T (paper: K=40, T=0.4)
  2. Executes each generated program
  3. Takes majority vote over numeric answers

Paper reports PoT+SC outperforms CoT+SC by ~10% on average.
We default to K=10 (gives ~90% of the benefit at 25% of the API cost).

Usage:
    from extensions.self_consistency import self_consistent_pot
    answer = self_consistent_pot(question, exemplar_prompt_fn, model="gpt-4o-mini", k=10)
"""

import collections
import math
from typing import Callable, Optional

import config
from utils.executor   import execute_program
from utils.llm_client import get_completion


def self_consistent_pot(
    question: str,
    build_prompt_fn: Callable[[str], str],
    model: Optional[str] = None,
    k: int = None,
    temperature: float = 0.4,
    system_prompt: str = "",
) -> dict:
    """
    Run self-consistent PoT on a single question.

    Parameters
    ----------
    question        : The math question string.
    build_prompt_fn : Function that takes a question and returns the full prompt.
    model           : LLM model name.
    k               : Number of samples (default: config.SC_SAMPLES).
    temperature     : Sampling temperature (paper uses 0.4).
    system_prompt   : Optional system instruction.

    Returns
    -------
    dict with keys:
      answer      — majority vote answer
      all_answers — list of all k answers
      codes       — list of all k programs
      vote_counts — Counter of answer → count
      confidence  — fraction of samples that agree with majority
    """
    model = model or config.DEFAULT_MODEL
    k     = k     or config.SC_SAMPLES

    prompt = build_prompt_fn(question)

    # Get k completions in one API call (or loop for Anthropic)
    completions = get_completion(
        prompt=prompt,
        system_prompt=system_prompt,
        model=model,
        temperature=temperature,
        n=k,
    )

    # Execute all programs
    answers = []
    codes   = []
    errors  = []
    for code in completions:
        ans, err = execute_program(code, timeout=config.CODE_TIMEOUT_SECS)
        answers.append(ans)
        codes.append(code)
        errors.append(err)

    # Filter out None / error answers
    valid = [(a, c) for a, c in zip(answers, codes) if a is not None]
    if not valid:
        return {
            "answer":      None,
            "all_answers": answers,
            "codes":       codes,
            "vote_counts": {},
            "confidence":  0.0,
        }

    valid_answers = [a for a, _ in valid]

    # Majority vote: bucket answers within tolerance
    majority_answer = _majority_vote(valid_answers)
    vote_counts     = _count_votes(valid_answers)
    confidence      = max(vote_counts.values()) / k if vote_counts else 0.0

    return {
        "answer":      majority_answer,
        "all_answers": answers,
        "codes":       codes,
        "vote_counts": vote_counts,
        "confidence":  confidence,
        "errors":      errors,
    }


def _majority_vote(answers: list, tol: float = 1e-3) -> Optional[float]:
    """
    Find the most common answer, treating values within `tol` as equal.
    Uses a simple O(n²) bucketing approach suitable for small k.
    """
    if not answers:
        return None

    # Build clusters
    clusters: list[list] = []
    for ans in answers:
        placed = False
        for cluster in clusters:
            representative = cluster[0]
            try:
                if _approx_equal(ans, representative, tol):
                    cluster.append(ans)
                    placed = True
                    break
            except (TypeError, ValueError):
                if str(ans) == str(representative):
                    cluster.append(ans)
                    placed = True
                    break
        if not placed:
            clusters.append([ans])

    # Pick largest cluster
    biggest = max(clusters, key=len)
    # Return the median value from the biggest cluster
    try:
        sorted_cluster = sorted(biggest)
        return sorted_cluster[len(sorted_cluster) // 2]
    except TypeError:
        return biggest[0]


def _count_votes(answers: list, tol: float = 1e-3) -> dict:
    """Return a dict mapping str(rounded_answer) → count."""
    counts: dict = {}
    for ans in answers:
        key = _round_key(ans)
        counts[key] = counts.get(key, 0) + 1
    return counts


def _approx_equal(a, b, tol: float) -> bool:
    try:
        fa, fb = float(a), float(b)
        if fa == 0 and fb == 0:
            return True
        return math.isclose(fa, fb, rel_tol=tol, abs_tol=tol)
    except (TypeError, ValueError):
        return str(a) == str(b)


def _round_key(ans) -> str:
    try:
        return str(round(float(ans), 4))
    except (TypeError, ValueError):
        return str(ans)
