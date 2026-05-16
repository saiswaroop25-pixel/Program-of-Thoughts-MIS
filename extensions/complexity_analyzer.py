"""
extensions/complexity_analyzer.py
───────────────────────────────────
🆕 NEW — not in the original paper.

Complexity-Aware Method Router.

The paper's breakdown analysis (Figure 6) shows:
  • PoT >> CoT on: iterative, polynomial/linear equations, symbolic, combinatorics
  • PoT ≈ CoT on: arithmetic, probability, geometry

This module analyses a question BEFORE calling the LLM and routes to the
most appropriate method:
  → PoT-only         : for computation-heavy problems
  → PoT + CoT hybrid : for problems needing post-computation reasoning
  → CoT-only         : for geometry/commonsense reasoning
  → PoT + Verify     : for algebraic problems (adds SymPy double-check)

This makes the system adaptive rather than applying one method uniformly.

Usage:
    from extensions.complexity_analyzer import analyse, route_method
    analysis = analyse("What is the 50th Fibonacci number?")
    print(analysis)
    # {"category": "iterative", "recommended": "pot_only", "confidence": 0.95}
"""

import re
from dataclasses import dataclass
from typing import Literal

# ── Problem categories (matches paper's Figure 6 breakdown) ──────────────────
Category = Literal[
    "arithmetic",
    "linear_equation",
    "polynomial_equation",
    "iterative",
    "combinatorics",
    "probability",
    "geometry",
    "symbolic",
    "financial",
    "unknown",
]

Method = Literal[
    "pot_only",
    "pot_verify",      # PoT + SymPy verification
    "pot_cot_hybrid",  # PoT computation then CoT reasoning
    "cot_only",
]


@dataclass
class ComplexityAnalysis:
    category:    Category
    recommended: Method
    confidence:  float         # 0–1
    signals:     list[str]     # Which patterns triggered this classification
    complexity_score: int      # 1 (easy) to 5 (hard)
    notes:       str


# ── Keyword/pattern rules ─────────────────────────────────────────────────────

_RULES: list[dict] = [
    # Iterative (loops needed) → PoT strongly preferred
    {
        "category": "iterative",
        "method":   "pot_only",
        "patterns": [
            r"\bevery\s+(?:day|year|month|week|hour)\b",
            r"\bfor\s+each\b",
            r"\brepeat\w*\b",
            r"\bsequence\b",
            r"\bfibonacci\b",
            r"\buntil\b.*\bpay\w*\b",
            r"\bcompound\s+interest\b.*\byear\b",
        ],
        "complexity": 4,
    },
    # Polynomial / cubic / quadratic → PoT + Verify (SymPy shines here)
    {
        "category": "polynomial_equation",
        "method":   "pot_verify",
        "patterns": [
            r"\bquadratic\b",
            r"\bcubic\b",
            r"\bpolynomial\b",
            r"x\^[23]",
            r"\bsquare\s+root\b",
            r"\bsolve\s+for\b",
            r"\bequation\b.*\bsolution\b",
        ],
        "complexity": 5,
    },
    # Linear equations → PoT + Verify
    {
        "category": "linear_equation",
        "method":   "pot_verify",
        "patterns": [
            r"\blet\s+x\s+be\b",
            r"\bif\s+x\b",
            r"\bsimple\s+interest\b",
            r"\brate\s+of\s+interest\b",
            r"\bcontribute\b.*\beach\b",
            r"\bspeed\b.*\bstream\b",
            r"\bboat\b.*\bstream\b",
        ],
        "complexity": 4,
    },
    # Combinatorics → PoT (factorial, combinations)
    {
        "category": "combinatorics",
        "method":   "pot_only",
        "patterns": [
            r"\bchoose\b",
            r"\bcombination\b",
            r"\bpermutation\b",
            r"\barrangement\b",
            r"\bfactorial\b",
            r"\bways\s+(?:can|to)\b",
            r"\bhow\s+many\s+ways\b",
        ],
        "complexity": 4,
    },
    # Financial / large numbers → PoT (CoT fails on large number arithmetic)
    {
        "category": "financial",
        "method":   "pot_only",
        "patterns": [
            r"\bmillion\b",
            r"\bbillion\b",
            r"\brevenue\b",
            r"\bprofit\s+margin\b",
            r"\bEBITDA\b",
            r"\bbalance\s+sheet\b",
            r"\bquarterly\b",
        ],
        "complexity": 3,
    },
    # MCQ with time conversion → PoT + CoT hybrid
    {
        "category": "arithmetic",
        "method":   "pot_cot_hybrid",
        "patterns": [
            r"\bwhat\s+time\b",
            r"\bHH:MM\b",
            r"\bpm\b.*\boption\b",
            r"\bAnswer\s+option\b",
            r"\bA\)\s*\d+\s+hour",
        ],
        "complexity": 3,
    },
    # Geometry / spatial → CoT (PoT doesn't help much per paper Figure 6)
    {
        "category": "geometry",
        "method":   "cot_only",
        "patterns": [
            r"\bperimeter\b",
            r"\barea\s+of\s+(?:a\s+)?(?:circle|triangle|square|rectangle)\b",
            r"\bangle\b",
            r"\bparallel\b",
            r"\bpythagor\w+\b",
            r"\bsimilar\s+triangles\b",
        ],
        "complexity": 3,
    },
    # Probability → CoT or PoT equally (per paper)
    {
        "category": "probability",
        "method":   "pot_only",
        "patterns": [
            r"\bprobability\b",
            r"\blikelihood\b",
            r"\bchance\b",
            r"\bdice\b",
            r"\bcard\b.*\bdraw\b",
        ],
        "complexity": 3,
    },
]


def analyse(question: str) -> ComplexityAnalysis:
    """
    Analyse a math question and return a complexity analysis + method recommendation.

    Parameters
    ----------
    question : The question text.

    Returns
    -------
    ComplexityAnalysis dataclass.
    """
    question_lower = question.lower()
    matched_signals: list[str] = []
    matched_rule = None
    max_matches = 0

    for rule in _RULES:
        matches = []
        for pattern in rule["patterns"]:
            if re.search(pattern, question_lower):
                matches.append(pattern)
        if len(matches) > max_matches:
            max_matches   = len(matches)
            matched_rule  = rule
            matched_signals = matches

    if matched_rule is None or max_matches == 0:
        # Default to PoT-only for unknown (it's usually safe)
        return ComplexityAnalysis(
            category="unknown",
            recommended="pot_only",
            confidence=0.4,
            signals=[],
            complexity_score=_estimate_complexity(question),
            notes="No specific pattern matched; defaulting to PoT.",
        )

    confidence = min(0.5 + max_matches * 0.15, 0.95)

    return ComplexityAnalysis(
        category=matched_rule["category"],
        recommended=matched_rule["method"],
        confidence=confidence,
        signals=matched_signals,
        complexity_score=matched_rule["complexity"],
        notes=_build_note(matched_rule["category"], matched_rule["method"]),
    )


def route_method(question: str) -> tuple[Method, ComplexityAnalysis]:
    """
    Convenience function: returns (recommended_method, analysis).
    """
    analysis = analyse(question)
    return analysis.recommended, analysis


def _estimate_complexity(question: str) -> int:
    """Heuristic complexity from question length and number count."""
    words   = len(question.split())
    numbers = len(re.findall(r"\d+", question))
    if words < 30 and numbers < 3:
        return 1
    if words < 60:
        return 2
    if words < 100:
        return 3
    return 4


def _build_note(category: Category, method: Method) -> str:
    notes = {
        ("iterative", "pot_only"):
            "Iterative computation: PoT handles loops efficiently. CoT would need 50+ steps.",
        ("polynomial_equation", "pot_verify"):
            "Algebraic equation: PoT with SymPy solve(). Verification adds correctness guarantee.",
        ("linear_equation", "pot_verify"):
            "Linear system: PoT + SymPy, then verify the solution satisfies original equations.",
        ("combinatorics", "pot_only"):
            "Combinatorics: factorial/nCr via math library. PoT is exact; CoT prone to errors.",
        ("financial", "pot_only"):
            "Financial computation: large numbers → LLM arithmetic fails. PoT with Python floats.",
        ("geometry", "cot_only"):
            "Geometric reasoning: spatial understanding benefits more from CoT rationale.",
        ("probability", "pot_only"):
            "Probability: counting + division — PoT handles exactly.",
    }
    return notes.get((category, method), f"Category: {category} → {method}")
