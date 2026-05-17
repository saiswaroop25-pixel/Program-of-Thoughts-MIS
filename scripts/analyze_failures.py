"""Summarise execution and prediction failures in saved JSONL outputs."""

from __future__ import annotations

import argparse
import html
import json
import re
from collections import Counter
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="JSONL output file to analyse")
    parser.add_argument("--out", default="", help="Optional markdown output path")
    parser.add_argument("--plot", default="", help="Optional SVG plot output path")
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

    if args.plot:
        plot_path = Path(args.plot)
        plot_path.parent.mkdir(exist_ok=True)
        plot_path.write_text(build_failure_plot(path, rows), encoding="utf-8")
        print(f"Wrote {plot_path}")


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


def build_failure_plot(path: Path, rows: list[dict]) -> str:
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

    summary = [
        ("Execution errors", len(exec_errors), "#7c5aa6"),
        ("Missing predictions", len(missing), "#b24a54"),
        ("Missing numeric answers", len(none_numeric), "#d08a35"),
        ("Wrong non-empty predictions", len(wrong), "#5b7db1"),
    ]
    error_types = _error_counter(exec_errors).most_common(8)

    width = 980
    height = 620
    parts = [
        _svg_open(width, height),
        _text(width / 2, 30, f"Failure Analysis: {path.name}", 18, "middle", bold=True),
        _text(width / 2, 52, f"Total examples: {total}", 12, "middle"),
    ]

    parts.extend(_bar_section(
        x=70,
        y=95,
        width=390,
        height=330,
        title="Failure Category Counts",
        values=summary,
        ymax=max(total, 1),
        value_suffix="",
        show_percent=True,
        total=total,
    ))

    typed_values = [(label, count, "#2f7f6f") for label, count in error_types]
    if not typed_values:
        typed_values = [("No execution errors", 0, "#2f7f6f")]
    parts.extend(_bar_section(
        x=540,
        y=95,
        width=370,
        height=330,
        title="Top Execution Error Types",
        values=typed_values,
        ymax=max([count for _, count, _ in typed_values] + [1]),
        value_suffix="",
        show_percent=False,
        total=total,
        max_label_len=34,
    ))

    parts.extend([
        _text(70, 485, "Interpretation", 14, "start", bold=True),
        _wrapped_text(
            70,
            510,
            "Most failures come from program generation rather than arithmetic. "
            "The model emits invalid SymPy code, empty solution indexing, or expressions "
            "that cannot be converted into a numeric multiple-choice option.",
            105,
            14,
        ),
        "</svg>",
    ])
    return "\n".join(parts)


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


def _bar_section(
    x: float,
    y: float,
    width: float,
    height: float,
    title: str,
    values: list[tuple[str, int, str]],
    ymax: int,
    value_suffix: str = "",
    show_percent: bool = False,
    total: int = 0,
    max_label_len: int = 24,
) -> list[str]:
    left_pad = 150
    top_pad = 45
    row_h = 34
    chart_w = width - left_pad - 20
    parts = [
        _text(x, y, title, 14, "start", bold=True),
        f'<line x1="{x}" y1="{y + 16}" x2="{x + width}" y2="{y + 16}" stroke="#ddd"/>',
    ]
    for i, (label, count, color) in enumerate(values):
        row_y = y + top_pad + i * row_h
        bar_w = 0 if ymax <= 0 else chart_w * count / ymax
        display_label = _trim(label, max_label_len)
        value = f"{count}{value_suffix}"
        if show_percent and total:
            value += f" ({count / total * 100:.1f}%)"
        parts.extend([
            _text(x, row_y + 13, display_label, 10, "start"),
            _rect(x + left_pad, row_y, chart_w, 18, "#f0f2f4"),
            _rect(x + left_pad, row_y, bar_w, 18, color),
            _text(x + left_pad + chart_w + 8, row_y + 13, value, 10, "start"),
        ])
    return parts


def _svg_open(width: int, height: int) -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}"><rect width="100%" height="100%" fill="white"/>'
    )


def _rect(x: float, y: float, width: float, height: float, fill: str) -> str:
    return f'<rect x="{x:.1f}" y="{y:.1f}" width="{max(width, 0):.1f}" height="{height:.1f}" fill="{fill}" rx="2"/>'


def _text(x: float, y: float, value: str, size: int, anchor: str, bold: bool = False) -> str:
    weight = "700" if bold else "400"
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" text-anchor="{anchor}" '
        f'font-family="Arial, sans-serif" font-size="{size}" font-weight="{weight}" '
        f'fill="#222">{html.escape(str(value))}</text>'
    )


def _wrapped_text(x: float, y: float, value: str, max_chars: int, line_h: int) -> str:
    words = value.split()
    lines = []
    current = ""
    for word in words:
        if len(current) + len(word) + 1 > max_chars:
            lines.append(current)
            current = word
        else:
            current = f"{current} {word}".strip()
    if current:
        lines.append(current)
    return "\n".join(_text(x, y + i * line_h, line, 11, "start") for i, line in enumerate(lines))


def _trim(value: str, max_len: int) -> str:
    return value if len(value) <= max_len else value[: max_len - 3] + "..."


if __name__ == "__main__":
    main()
