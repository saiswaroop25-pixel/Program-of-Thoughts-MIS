"""
utils/data_loader.py
────────────────────
Loads benchmark datasets from HuggingFace — no manual download needed.

Datasets:
  GSM8K   : grade school math word problems
  AQuA    : algebraic MCQ questions
  SVAMP   : simple variation math word problems
  MultiArith: multi-step arithmetic word problems
"""

import json
import os
from typing import Optional

from datasets import load_dataset

import config


def load_gsm8k(split: str = "test",
               start: int = 0,
               end: int = -1) -> list[dict]:
    """
    Load GSM8K test set.
    Returns list of {"question": str, "answer": float}
    """
    local_path = os.path.join(config.DATA_DIR, "gsm8K.json")
    if split == "test" and os.path.exists(local_path):
        with open(local_path, encoding="utf-8") as f:
            items = []
            for row in json.load(f):
                answer = row.get("answer")
                if isinstance(answer, str) and "####" in answer:
                    answer = answer.split("####")[-1].strip().replace(",", "")
                try:
                    answer = float(answer)
                except (TypeError, ValueError):
                    pass
                items.append({"question": row["question"], "answer": answer})
            return _slice(items, start, end)

    if os.getenv("POT_ALLOW_DATASET_DOWNLOAD", "1") != "1":
        raise RuntimeError(
            "GSM8K data is not available locally. Put data/gsm8K.json in the project "
            "or set POT_ALLOW_DATASET_DOWNLOAD=1 to download it from HuggingFace."
        )

    ds = load_dataset("gsm8k", "main", split=split)
    items = []
    for row in ds:
        # Answer is at the end after "####"
        answer_str = row["answer"].split("####")[-1].strip().replace(",", "")
        try:
            answer = float(answer_str)
        except ValueError:
            answer = answer_str
        items.append({"question": row["question"], "answer": answer})

    return _slice(items, start, end)


def load_aqua(split: str = "test",
              start: int = 0,
              end: int = -1) -> list[dict]:
    """
    Load AQuA-RAT test set.
    Returns list of {"question": str, "options": list[str], "answer": str}
    """
    ds = load_dataset("aqua_rat", "raw", split=split)
    items = []
    for row in ds:
        items.append({
            "question": row["question"],
            "options":  row["options"],
            "answer":   row["correct"],  # e.g. "B"
            "rationale": row.get("rationale", ""),
        })
    return _slice(items, start, end)


def load_svamp(start: int = 0, end: int = -1) -> list[dict]:
    """
    Load SVAMP test set.
    Returns list of {"question": str, "answer": float}
    """
    ds = load_dataset("ChilleD/SVAMP", split="test")
    items = []
    for row in ds:
        body   = row.get("Body", "")
        question_part = row.get("Question", "")
        question = f"{body} {question_part}".strip()
        items.append({
            "question": question,
            "answer":   float(row["Answer"]),
            "equation": row.get("Equation", ""),
        })
    return _slice(items, start, end)


def load_multiarith(start: int = 0, end: int = -1) -> list[dict]:
    """
    Load MultiArith test set.
    Returns list of {"question": str, "answer": float}
    """
    ds = load_dataset("ChilleD/MultiArith", split="test")
    items = []
    for row in ds:
        items.append({
            "question": row["question"],
            "answer":   float(row["final_ans"]),
        })
    return _slice(items, start, end)


def _slice(items: list, start: int, end: int) -> list:
    if end == -1:
        return items[start:]
    return items[start:end]
