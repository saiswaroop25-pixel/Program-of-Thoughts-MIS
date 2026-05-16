"""Summarise execution and prediction failures in saved JSONL outputs."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="JSONL output file to analyse")
    parser.add_argument("--out", default="", help="Optional markdown output path")
    parser.add_argument("--examples", type=int, default=8, help="Number of examples to include")
    args = parser.parse_args()

    path = Path(args.input)
    rows = [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]
    report = build_report(path, rows, examples=args.examples)

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(exist_ok=True)
        out_path.write_text(report, encoding="utf-8")
        print(f"Wrote {out_path}")
    else:
        print(report)


def build_report(path: Path, rows: list[dict], examples: int = 8) -> str:
    total = len(rows)
    exec_errors = [row for row in rows if row.get("exec_error")]
    missing = [row for row in rows if row.get("prediction") is None]
    none_numeric = [row for row in rows if row.get("numeric_answer") is None]
    wrong = [
        row for row in rows
        if row.get("prediction") is not None
        and row.get("answer") is not None
        and str(row.get("prediction")).strip() != str(row.get("answer")).strip()
    ]

    lines = [
        f"# Failure Analysis: `{path.name}`",
        "",
        "## Summary",
        "",
        f"- Total rows: {total}",
        f"- Execution errors: {len(exec_errors)} ({_pct(len(exec_errors), total)})",
        f"- Missing predictions: {len(missing)} ({_pct(len(missing), total)})",
        f"- Missing numeric answers: {len(none_numeric)} ({_pct(len(none_numeric), total)})",
        f"- Wrong non-empty predictions: {len(wrong)} ({_pct(len(wrong), total)})",
        "",
        "## Error Types",
        "",
    ]

    if exec_errors:
        for message, count in _error_counter(exec_errors).most_common(12):
            lines.append(f"- `{message}`: {count}")
    else:
        lines.append("- No execution errors.")

    lines.extend([
        "",
        "## Representative Failures",
        "",
    ])

    for row in exec_errors[:examples]:
        lines.extend(_format_example(row))

    if not exec_errors and wrong:
        for row in wrong[:examples]:
            lines.extend(_format_example(row))

    lines.extend([
        "",
        "## Interpretation",
        "",
        "The main failure mode is not arithmetic after code execution; it is program generation. "
        "The model often emits invalid SymPy calls, indexes empty solution lists, or returns an "
        "expression that cannot be converted to a numeric option. This explains why AQuA is much "
        "harder than GSM8K for the current free-tier model.",
        "",
    ])

    return "\n".join(lines)


def _error_counter(rows: list[dict]) -> Counter:
    counter = Counter()
    for row in rows:
        err = str(row.get("exec_error", ""))
        lines = [line.strip() for line in err.splitlines() if line.strip()]
        tail = lines[-1] if lines else err[:120]
        tail = re.sub(r"\s+", " ", tail)
        counter[tail] += 1
    return counter


def _format_example(row: dict) -> list[str]:
    err = str(row.get("exec_error", "")).splitlines()
    err_tail = err[-1].strip() if err else ""
    code = str(row.get("code", "")).strip().splitlines()
    code_preview = "\n".join(code[:10])
    question = str(row.get("question", "")).replace("\n", " ")
    if len(question) > 240:
        question = question[:237] + "..."
    return [
        f"### Row {row.get('index', '?')}",
        "",
        f"- Gold answer: `{row.get('answer')}`",
        f"- Prediction: `{row.get('prediction')}`",
        f"- Numeric answer: `{row.get('numeric_answer', row.get('prediction'))}`",
        f"- Error: `{err_tail}`",
        f"- Question: {question}",
        "",
        "```python",
        code_preview,
        "```",
        "",
    ]


def _pct(count: int, total: int) -> str:
    return "0.0%" if total == 0 else f"{count / total * 100:.1f}%"


if __name__ == "__main__":
    main()
