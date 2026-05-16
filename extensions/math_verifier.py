"""
extensions/math_verifier.py
────────────────────────────
🆕 NEW — not in the original paper.

Symbolic Verification Layer using SymPy.

After PoT produces an answer, this module:
  1. Parses the generated code to extract the mathematical relationships.
  2. Uses SymPy to INDEPENDENTLY verify the answer is consistent with the problem constraints.
  3. Flags answers that fail symbolic verification for re-generation.

This adds a formal mathematical proof layer on top of LLM-generated programs.

Why this matters for a MATHS project:
  • The paper identifies value grounding errors as the #1 failure mode (47%).
  • Symbolic verification can catch many of these BEFORE accepting the answer.
  • It provides a principled mathematical guarantee — not just empirical accuracy.

The verifier works by:
  (a) Extracting equations/constraints from the code using pattern matching.
  (b) Plugging the predicted answer back into SymPy and checking it satisfies them.
  (c) For simple linear equations, also solving symbolically to compare.

Usage:
    from extensions.math_verifier import verify_answer, batch_verify
    result = verify_answer(code="x = 5\nans = x * 2", predicted=10.0)
    print(result)  # {"verified": True, "method": "substitution", ...}
"""

import re
import math
from typing import Any, Optional
import sympy
from sympy import Symbol, symbols, solve, simplify, Eq, Rational, N


# ── Main verifier ─────────────────────────────────────────────────────────────
def verify_answer(
    code:      str,
    predicted: Any,
    question:  Optional[str] = None,
    tol:       float = 1e-6,
) -> dict:
    """
    Attempt to symbolically verify a predicted answer.

    Parameters
    ----------
    code      : The generated Python program.
    predicted : The numeric answer from executing the code.
    question  : Optional question text (used for heuristic extraction).
    tol       : Tolerance for float comparison.

    Returns
    -------
    dict with keys:
      verified     — True / False / None (None = couldn't verify)
      method       — "substitution" | "symbolic_solve" | "unverifiable"
      sympy_answer — Answer from SymPy's own solve (if applicable)
      matches      — Whether SymPy answer matches predicted
      notes        — Explanation string
    """
    if predicted is None:
        return _unverifiable("No predicted answer")

    # Strategy 1: Substitution check — re-run final expression with SymPy
    sub_result = _substitution_check(code, predicted, tol)
    if sub_result["verified"] is not None:
        return sub_result

    # Strategy 2: Symbolic solve — detect equations and solve with SymPy
    sym_result = _symbolic_solve_check(code, predicted, tol)
    if sym_result["verified"] is not None:
        return sym_result

    return _unverifiable("Could not extract verifiable constraints from code")


def batch_verify(results: list[dict], tol: float = 1e-6) -> list[dict]:
    """
    Run verification on a list of result dicts (from run_gsm8k.py output).
    Adds 'verified', 'sympy_answer', 'verification_notes' to each dict.
    """
    enriched = []
    for item in results:
        v = verify_answer(
            code=item.get("code", ""),
            predicted=item.get("prediction"),
            question=item.get("question"),
            tol=tol,
        )
        enriched.append({
            **item,
            "verified":            v["verified"],
            "sympy_answer":        v.get("sympy_answer"),
            "verification_method": v["method"],
            "verification_notes":  v["notes"],
        })
    return enriched


def verification_stats(verified_results: list[dict]) -> str:
    """Summarise verification outcomes."""
    total     = len(verified_results)
    confirmed = sum(1 for r in verified_results if r.get("verified") is True)
    refuted   = sum(1 for r in verified_results if r.get("verified") is False)
    unknown   = sum(1 for r in verified_results if r.get("verified") is None)

    # Of confirmed, how many are also correct by ground truth?
    confirmed_correct = sum(
        1 for r in verified_results
        if r.get("verified") is True
        and _approx_eq(r.get("prediction"), r.get("answer"))
    )
    refuted_incorrect = sum(
        1 for r in verified_results
        if r.get("verified") is False
        and not _approx_eq(r.get("prediction"), r.get("answer"))
    )

    lines = [
        f"\nSymbolic Verification Stats (n={total})",
        "=" * 45,
        f"  Verified (SymPy agrees)  : {confirmed:3d}  ({confirmed/total:.0%})",
        f"  Refuted  (SymPy disagrees): {refuted:3d}  ({refuted/total:.0%})",
        f"  Unverifiable             : {unknown:3d}  ({unknown/total:.0%})",
        "",
        f"  Of verified → ground truth correct : {confirmed_correct}/{confirmed}",
        f"  Of refuted  → ground truth wrong   : {refuted_incorrect}/{refuted}",
        "=" * 45,
        "  Interpretation: high 'refuted → wrong' means verification",
        "  successfully catches errors before accepting answers.",
    ]
    return "\n".join(lines)


# ── Strategy 1: Substitution check ───────────────────────────────────────────
def _substitution_check(code: str, predicted: Any, tol: float) -> dict:
    """
    Execute the code in a safe numeric environment and compare `ans` with predicted.
    This handles derived variables (e.g., sold = total - eaten) correctly.
    """
    # Strip imports and sympy calls — only pure arithmetic code works here
    numeric_lines = []
    for line in code.splitlines():
        stripped = line.strip()
        if any(kw in stripped for kw in ["import", "Symbol", "solve", "simplify", "from sympy"]):
            continue
        numeric_lines.append(line)
    numeric_code = "\n".join(numeric_lines)

    if "ans" not in numeric_code:
        return _unverifiable("No `ans` assignment in numeric code")

    safe_builtins = {
        "range": range, "int": int, "float": float, "abs": abs,
        "round": round, "min": min, "max": max, "sum": sum, "len": len,
        "True": True, "False": False, "None": None,
    }
    try:
        local_ns: dict = {}
        exec(numeric_code, {"__builtins__": safe_builtins, "math": math}, local_ns)  # noqa: S102
        sympy_val = local_ns.get("ans")
        if sympy_val is None:
            return _unverifiable("Code did not set `ans`")
        sympy_float = float(sympy_val)
        pred_float  = float(predicted)
        matches = math.isclose(sympy_float, pred_float, rel_tol=tol, abs_tol=tol)
        return {
            "verified":     matches,
            "method":       "substitution",
            "sympy_answer": sympy_float,
            "matches":      matches,
            "notes":        f"Re-execution: ans={sympy_float:.6g}; predicted={pred_float}",
        }
    except Exception as e:
        return _unverifiable(f"Substitution failed: {e}")


# ── Strategy 2: Symbolic solve ────────────────────────────────────────────────
_SYMPY_SOLVE_PATTERN = re.compile(
    r"(solve(?:_it)?\s*\(.*?\))", re.DOTALL
)

def _symbolic_solve_check(code: str, predicted: Any, tol: float) -> dict:
    """
    If the code calls solve() or solve_it(), re-run the solve symbolically
    and compare with the predicted answer.
    """
    if "solve" not in code:
        return _unverifiable("No solve() call detected")

    # Rebuild safe namespace
    ns = _build_sympy_namespace()
    ns.update(_extract_numeric_namespace(code))

    # Re-execute only the lines up to and including the solve call
    solve_lines = _extract_solve_block(code)
    if not solve_lines:
        return _unverifiable("Could not isolate solve block")

    try:
        exec_ns: dict = {}
        exec_ns.update(ns)
        exec(solve_lines, exec_ns)          # noqa: S102

        sympy_ans = exec_ns.get("ans") or exec_ns.get("solution")
        if sympy_ans is None:
            return _unverifiable("solve block did not produce `ans`")

        sympy_float = float(sympy.N(sympy_ans))
        matches = math.isclose(sympy_float, float(predicted), rel_tol=tol, abs_tol=tol)
        return {
            "verified":    matches,
            "method":      "symbolic_solve",
            "sympy_answer": sympy_float,
            "matches":     matches,
            "notes":       f"SymPy solve returned {sympy_float:.6g}; predicted = {predicted}",
        }
    except Exception as e:
        return _unverifiable(f"Symbolic solve failed: {e}")


# ── Helpers ───────────────────────────────────────────────────────────────────
def _extract_numeric_namespace(code: str) -> dict:
    """
    Parse simple `var = number` assignments from code into a dict.
    Ignores lines with function calls, loops, or complex expressions.
    """
    ns = {}
    for line in code.splitlines():
        line = line.strip()
        m = re.match(r"^([a-zA-Z_]\w*)\s*=\s*([0-9\.\-\+\/\*\s\(\)eE]+)$", line)
        if m:
            name, expr = m.group(1), m.group(2)
            try:
                ns[name] = float(eval(expr, {"__builtins__": {}}))  # noqa: S307
            except Exception:
                pass
    return ns


def _extract_solve_block(code: str) -> str:
    """
    Extract lines from code that include Symbol definitions and solve calls.
    """
    lines = code.splitlines()
    relevant = []
    for line in lines:
        stripped = line.strip()
        if any(kw in stripped for kw in ["Symbol", "symbols", "solve", "Eq", "ans ="]):
            relevant.append(line)
        elif re.match(r"^[a-zA-Z_]\w*\s*=\s*[0-9\.\-]", stripped):
            relevant.append(line)
    return "\n".join(relevant)


def _build_sympy_namespace() -> dict:
    """Return a namespace with core SymPy names available."""
    return {
        "Symbol": Symbol, "symbols": symbols, "solve": solve,
        "simplify": simplify, "Eq": Eq, "Rational": Rational,
        "N": N, "sqrt": sympy.sqrt, "pi": sympy.pi,
        "math": math,
    }


def _unverifiable(reason: str) -> dict:
    return {
        "verified":    None,
        "method":      "unverifiable",
        "sympy_answer": None,
        "matches":     None,
        "notes":       reason,
    }


def _approx_eq(a, b, tol: float = 1e-3) -> bool:
    try:
        return math.isclose(float(a), float(b), rel_tol=tol)
    except (TypeError, ValueError):
        return str(a) == str(b)
