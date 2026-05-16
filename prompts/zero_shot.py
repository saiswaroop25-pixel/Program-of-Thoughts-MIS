"""
prompts/zero_shot.py
─────────────────────
Zero-shot PoT and CoT prompt templates.

Key difference from few-shot:
  • No exemplar demonstrations
  • Uses an instruction to guide the LLM
  • Zero-shot PoT is simpler than zero-shot CoT — no extra extraction step needed
    because the answer is always stored in the `ans` variable by convention.
"""

# ── Zero-Shot PoT ─────────────────────────────────────────────────────────────
ZS_POT_SYSTEM = (
    "You are an expert mathematician and Python programmer. "
    "When given a math problem, write a clean Python program to solve it. "
    "Use descriptive variable names that reflect the problem context. "
    "Always store the final answer in a variable called `ans`. "
    "Do not use any markdown formatting — output only raw Python code."
)

ZS_POT_TEMPLATE = """\
# Answer this question by implementing a solver() function.
# Let's write a Python program step by step, then return the answer.
# Firstly, we need to define the following variables:

Question: {question}

def solver():
    # Let's write a Python program step by step, and then return the answer"""


def build_zero_shot_pot(question: str) -> tuple[str, str]:
    """
    Returns (system_prompt, user_prompt) for zero-shot PoT.
    """
    user = ZS_POT_TEMPLATE.format(question=question)
    return ZS_POT_SYSTEM, user


# ── Zero-Shot CoT (baseline) ──────────────────────────────────────────────────
ZS_COT_SYSTEM = (
    "You are an expert mathematician. "
    "Solve the following math problem step by step, showing your reasoning. "
    "At the end of your solution, write 'The answer is: X' where X is the number."
)

ZS_COT_TEMPLATE = """\
Question: {question}

Let's think step by step:"""


def build_zero_shot_cot(question: str) -> tuple[str, str]:
    """
    Returns (system_prompt, user_prompt) for zero-shot CoT.
    """
    user = ZS_COT_TEMPLATE.format(question=question)
    return ZS_COT_SYSTEM, user


# ── CoT answer extraction ─────────────────────────────────────────────────────
import re

def extract_cot_answer(text: str) -> float | None:
    """
    Extract final numeric answer from a CoT response.
    Looks for patterns like:
      "The answer is: 42"  |  "= 42"  |  "42." at end
    """
    # Try "The answer is: X" pattern
    m = re.search(r"[Tt]he answer is[:\s]+([+-]?\d[\d,\.]*)", text)
    if m:
        return _parse(m.group(1))

    # Try "#### X" (GSM8K rationale format)
    m = re.search(r"####\s*([+-]?\d[\d,\.]*)", text)
    if m:
        return _parse(m.group(1))

    # Try last number in text
    nums = re.findall(r"[+-]?\d[\d,\.]*", text)
    if nums:
        return _parse(nums[-1])

    return None


def _parse(s: str) -> float | None:
    try:
        return float(s.replace(",", ""))
    except (ValueError, AttributeError):
        return None