"""
extensions/pot_cot_hybrid.py
─────────────────────────────
PoT → CoT Multi-Stage Pipeline (Section 2.3 of the paper, Figure 4).

For problems that require BOTH:
  (a) precise computation  → handled by PoT (Python interpreter)
  (b) semantic reasoning   → handled by CoT (e.g. convert 2.05 hours → "1:03 PM")

The pipeline:
  1. Run PoT to get an intermediate numeric result.
  2. If the LLM signals "keep prompting" (or we detect a non-terminal result),
     feed the result back into a CoT prompt for final reasoning.
  3. Otherwise return the PoT answer directly.

The paper only needed this for AQuA (time/option formatting).
We generalise it to any problem type.

Usage:
    from extensions.pot_cot_hybrid import hybrid_pot_cot
    result = hybrid_pot_cot(question, options=["A)1 hour", ...], model="gpt-4o-mini")
"""

import re
from typing import Optional

import config
from utils.executor   import execute_program
from utils.llm_client import get_completion
from prompts.zero_shot import extract_cot_answer


# ── Stage 1: PoT ──────────────────────────────────────────────────────────────
POT_STAGE1_SYSTEM = (
    "You are a Python programmer solving math problems. "
    "Write Python code to compute the intermediate result. "
    "If the problem requires more reasoning after computation (e.g. choosing from "
    "multiple choice options, formatting time), end your code with: "
    "# keep_prompting\n"
    "Otherwise just store the final answer in `ans`."
)

POT_STAGE1_TEMPLATE = "Question: {question}\n\n# Python code:"


# ── Stage 2: CoT refinement ───────────────────────────────────────────────────
COT_STAGE2_SYSTEM = (
    "You are a math expert. You have been given a question and a computed intermediate "
    "result from a Python program. Use this to determine the final answer. "
    "Write 'The answer is: X' at the end."
)

COT_STAGE2_TEMPLATE = """\
Question: {question}

According to the computation: {intermediate_name} = {intermediate_value}

Using this result, what is the final answer?
{options_str}

Let's reason:"""


def hybrid_pot_cot(
    question:  str,
    options:   Optional[list[str]] = None,
    model:     Optional[str] = None,
    shots:     int = 0,            # 0 = zero-shot
) -> dict:
    """
    Run the PoT → CoT hybrid pipeline on a single question.

    Returns
    -------
    dict with keys:
      pot_code         — generated Python code
      intermediate     — numeric result from executing the code
      needs_cot        — whether stage 2 was triggered
      cot_response     — CoT stage 2 response (if triggered)
      answer           — final answer
      exec_error       — any execution error from stage 1
    """
    model = model or config.DEFAULT_MODEL

    # ── Stage 1: PoT ──────────────────────────────────────────────────────────
    pot_prompt = POT_STAGE1_TEMPLATE.format(question=question)
    completions = get_completion(
        prompt=pot_prompt,
        system_prompt=POT_STAGE1_SYSTEM,
        model=model,
        temperature=0.0,
        n=1,
    )
    pot_code = completions[0] if completions else ""

    # Detect if model flagged "keep_prompting"
    needs_cot = "keep_prompting" in pot_code or _is_mcq(question, options)

    # Execute stage 1
    intermediate, exec_error = execute_program(
        _clean_keep_prompting(pot_code),
        timeout=config.CODE_TIMEOUT_SECS,
    )

    # If execution failed or answer is already clean, skip stage 2
    if exec_error and not needs_cot:
        return {
            "pot_code":    pot_code,
            "intermediate": None,
            "needs_cot":   False,
            "cot_response": None,
            "answer":      None,
            "exec_error":  exec_error,
        }

    if not needs_cot:
        return {
            "pot_code":    pot_code,
            "intermediate": intermediate,
            "needs_cot":   False,
            "cot_response": None,
            "answer":      intermediate,
            "exec_error":  exec_error,
        }

    # ── Stage 2: CoT refinement ───────────────────────────────────────────────
    options_str = ""
    if options:
        options_str = "Options: " + ", ".join(options)

    # Try to extract a meaningful variable name from the code
    intermediate_name = _extract_last_var(pot_code) or "result"

    cot_prompt = COT_STAGE2_TEMPLATE.format(
        question=question,
        intermediate_name=intermediate_name,
        intermediate_value=intermediate if intermediate is not None else "unknown",
        options_str=options_str,
    )

    cot_completions = get_completion(
        prompt=cot_prompt,
        system_prompt=COT_STAGE2_SYSTEM,
        model=model,
        temperature=0.0,
        n=1,
    )
    cot_response = cot_completions[0] if cot_completions else ""

    # Extract final answer from CoT
    final_answer = _extract_option(cot_response, options) or extract_cot_answer(cot_response)

    return {
        "pot_code":     pot_code,
        "intermediate": intermediate,
        "needs_cot":    True,
        "cot_response": cot_response,
        "answer":       final_answer,
        "exec_error":   exec_error,
    }


# ── Helpers ───────────────────────────────────────────────────────────────────
def _is_mcq(question: str, options: Optional[list]) -> bool:
    """Heuristic: is this a multiple-choice question?"""
    if options and len(options) > 1:
        return True
    return bool(re.search(r"\b[A-E]\)", question))


def _clean_keep_prompting(code: str) -> str:
    """Remove the # keep_prompting marker before execution."""
    return re.sub(r"#\s*keep_prompting", "", code)


def _extract_last_var(code: str) -> Optional[str]:
    """Extract the last assigned variable name (usually `ans`)."""
    matches = re.findall(r"^([a-zA-Z_][a-zA-Z0-9_]*)\s*=", code, re.MULTILINE)
    return matches[-1] if matches else None


def _extract_option(text: str, options: Optional[list]) -> Optional[str]:
    """Find a matching option letter in the CoT response."""
    if not options:
        return None
    for opt in options:
        letter = opt[0] if opt else ""
        if re.search(rf"\b{letter}\b", text):
            return letter
    return None
