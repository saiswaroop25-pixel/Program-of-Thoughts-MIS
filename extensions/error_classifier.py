"""
extensions/error_classifier.py
────────────────────────────────
Automatic error analysis replicating Figure 7 of the paper.

The paper manually classified PoT failures into two types:
  • Type 1 — Value Grounding Error: correct logic, wrong variable values
  • Type 2 — Logic Generation Error: correct values, wrong computation logic

We automate this using an LLM judge, then produce a breakdown summary.

Paper findings on TAT-QA:
  • 47% value grounding errors
  • 33% logic generation errors
  • 15% both types
  •  5% actually correct (evaluation error)

Usage:
    from extensions.error_classifier import classify_errors, summarise_errors
    classifications = classify_errors(error_cases, model="gpt-4o-mini")
    print(summarise_errors(classifications))
"""

from typing import Optional
import json

import config
from utils.llm_client import get_completion


# Error type constants
TYPE_VALUE_GROUNDING = "value_grounding"
TYPE_LOGIC           = "logic_generation"
TYPE_BOTH            = "both"
TYPE_CORRECT         = "actually_correct"
TYPE_OTHER           = "other"


CLASSIFIER_SYSTEM = """\
You are an expert code reviewer analysing failures in LLM-generated Python programs for math problems.

Classify each failure into ONE of these categories:
  - value_grounding: The computation logic is correct, but one or more variable values are wrong
    (e.g., wrong number extracted from the problem, wrong unit, misread table entry).
  - logic_generation: The variable values are correct, but the computation logic is wrong
    (e.g., subtracted instead of added, wrong formula, missed a step).
  - both: Both value grounding AND logic errors are present.
  - actually_correct: The predicted answer looks wrong but is actually correct
    (evaluation/rounding issue).
  - other: Neither category applies clearly.

Respond with a JSON object: {"type": "...", "reason": "brief explanation"}
Only output the JSON, nothing else."""

CLASSIFIER_TEMPLATE = """\
Question: {question}

Ground Truth Answer: {reference}

Generated Python Code:
```python
{code}
```

Predicted Answer: {prediction}
Execution Error: {exec_error}

Classify this failure:"""


def classify_error(
    item: dict,
    model: Optional[str] = None,
) -> dict:
    """
    Classify a single error case.

    Parameters
    ----------
    item  : dict with keys: question, answer, code, prediction, exec_error
    model : LLM model name

    Returns
    -------
    Original item dict + added keys: error_type, error_reason
    """
    model = model or config.DEFAULT_MODEL

    prompt = CLASSIFIER_TEMPLATE.format(
        question=item.get("question", ""),
        reference=item.get("answer", ""),
        code=item.get("code", "(no code generated)"),
        prediction=item.get("prediction", "None"),
        exec_error=item.get("exec_error", "None") or "None",
    )

    completions = get_completion(
        prompt=prompt,
        system_prompt=CLASSIFIER_SYSTEM,
        model=model,
        temperature=0.0,
        n=1,
    )
    raw = completions[0] if completions else "{}"

    try:
        parsed = json.loads(raw.strip())
        error_type   = parsed.get("type", TYPE_OTHER)
        error_reason = parsed.get("reason", "")
    except (json.JSONDecodeError, AttributeError):
        error_type   = TYPE_OTHER
        error_reason = raw[:200]

    return {
        **item,
        "error_type":   error_type,
        "error_reason": error_reason,
    }


def classify_errors(
    error_cases: list[dict],
    model: Optional[str] = None,
    verbose: bool = True,
) -> list[dict]:
    """
    Classify a list of error cases.

    Parameters
    ----------
    error_cases : List of dicts (from compute_accuracy errors output).
    model       : LLM model name.
    verbose     : Print progress.

    Returns
    -------
    List of dicts with added error_type and error_reason fields.
    """
    results = []
    for i, item in enumerate(error_cases):
        if verbose:
            print(f"  Classifying error {i+1}/{len(error_cases)}...", end="\r")
        classified = classify_error(item, model=model)
        results.append(classified)

    if verbose:
        print()
    return results


def summarise_errors(classified: list[dict]) -> str:
    """
    Print a summary breakdown matching Figure 7 of the paper.
    """
    if not classified:
        return "No errors to summarise."

    counts = {
        TYPE_VALUE_GROUNDING: 0,
        TYPE_LOGIC:           0,
        TYPE_BOTH:            0,
        TYPE_CORRECT:         0,
        TYPE_OTHER:           0,
    }
    for item in classified:
        t = item.get("error_type", TYPE_OTHER)
        counts[t] = counts.get(t, 0) + 1

    total = len(classified)
    lines = [
        f"\nError Analysis (n={total})",
        "=" * 40,
        f"  Value Grounding Errors : {counts[TYPE_VALUE_GROUNDING]:3d}  ({counts[TYPE_VALUE_GROUNDING]/total:.0%})",
        f"  Logic Generation Errors: {counts[TYPE_LOGIC]:3d}  ({counts[TYPE_LOGIC]/total:.0%})",
        f"  Both Error Types       : {counts[TYPE_BOTH]:3d}  ({counts[TYPE_BOTH]/total:.0%})",
        f"  Actually Correct       : {counts[TYPE_CORRECT]:3d}  ({counts[TYPE_CORRECT]/total:.0%})",
        f"  Other / Unclear        : {counts[TYPE_OTHER]:3d}  ({counts[TYPE_OTHER]/total:.0%})",
        "=" * 40,
        "\n  Paper's TAT-QA findings:",
        "  Value Grounding: 47% | Logic: 33% | Both: 15% | Correct: 5%",
    ]
    return "\n".join(lines)
