"""Create report-ready SVG plots from JSONL experiment outputs."""

from __future__ import annotations

import argparse
import html
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.evaluator import compute_accuracy


PAPER_BASELINES = {
    ("gsm8k", "cot_zs"): 40.5,
    ("gsm8k", "cot_fs"): 63.1,
    ("gsm8k", "pot_zs"): 57.0,
    ("gsm8k", "pot_fs"): 71.6,
    ("aqua", "pot_fs"): 54.1,
    ("multiarith", "pot_zs"): 92.2,
}

METHOD_LABELS = {
    "cot_zs": "CoT zero-shot",
    "cot_fs": "CoT few-shot",
    "pot_zs": "PoT zero-shot",
    "pot_fs": "PoT few-shot",
    "adaptive": "Adaptive PoT",
}

METHOD_ORDER = ["cot_zs", "cot_fs", "pot_zs", "pot_fs", "adaptive"]
COLORS = {
    "cot_zs": "#5b7db1",
    "cot_fs": "#88a758",
    "pot_zs": "#2f7f6f",
    "pot_fs": "#b65f3a",
    "adaptive": "#7c5aa6",
    "paper": "#d08a35",
    "missing": "#b24a54",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outputs", default="outputs")
    parser.add_argument("--plots", default="plots")
    args = parser.parse_args()

    output_dir = Path(args.outputs)
    plot_dir = Path(args.plots)
    plot_dir.mkdir(exist_ok=True)

    runs = [_summarise_run(path) for path in sorted(output_dir.glob("*.jsonl"))]
    runs = [run for run in runs if run is not None]
    if not runs:
        raise SystemExit(f"No scoreable JSONL files found in {output_dir}")

    _write_summary(runs, plot_dir / "results_summary.csv")
    _bar_chart(
        path=plot_dir / "accuracy_by_run.svg",
        title="Accuracy Across Available Runs",
        groups=[(_short_label(run), [(run["accuracy"], run["method"], "Project")]) for run in runs],
        ylabel="Accuracy (%)",
        ymax=105,
    )
    _plot_best_method_comparison(runs, plot_dir / "method_comparison_vs_paper.svg")
    _bar_chart(
        path=plot_dir / "execution_error_rates.svg",
        title="Execution and Missing-Prediction Rates",
        groups=[
            (
                _short_label(run),
                [
                    (run["exec_errors"] / run["n"] * 100, "adaptive", "Exec errors"),
                    (run["missing"] / run["n"] * 100, "missing", "Missing"),
                ],
            )
            for run in runs
        ],
        ylabel="Rate (%)",
        ymax=20,
        stacked=True,
    )
    _line_chart(runs, plot_dir / "cumulative_accuracy.svg")

    print(f"Wrote summary: {plot_dir / 'results_summary.csv'}")
    print(f"Wrote SVG plots in: {plot_dir}")


def _summarise_run(path: Path) -> dict | None:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if "answer" not in row:
                continue
            if "prediction" not in row and "executed" in row:
                row["prediction"] = row.get("executed")
            if "prediction" in row:
                rows.append(row)
    if not rows:
        return None

    dataset, method, model, start, end = _parse_filename(path.name)
    mode = "option" if dataset == "aqua" else "exact"
    metrics = compute_accuracy(rows, mode=mode)
    return {
        "file": path.name,
        "dataset": dataset,
        "method": method,
        "method_label": METHOD_LABELS.get(method, method),
        "model": model,
        "start": start,
        "end": end,
        "n": metrics["total"],
        "correct": metrics["correct"],
        "accuracy": metrics["accuracy"] * 100,
        "exec_errors": sum(1 for row in rows if row.get("exec_error")),
        "missing": sum(1 for row in rows if row.get("prediction") is None),
        "paper": PAPER_BASELINES.get((dataset, method)),
        "rows": rows,
    }


def _parse_filename(name: str) -> tuple[str, str, str, int | None, int | None]:
    stem = name.removesuffix(".jsonl")
    if stem.startswith("aqua"):
        dataset = "aqua"
    elif stem.startswith("multiarith"):
        dataset = "multiarith"
    else:
        dataset = "gsm8k"
    prefixes = {
        "gsm8k_cot_zs_": "cot_zs",
        "gsm8k_cot_fs_": "cot_fs",
        "gsm8k_zs_": "pot_zs",
        "gsm8k_fs_": "pot_fs",
        "gsm8k_adaptive_": "adaptive",
        "aqua_fs_": "pot_fs",
        "multiarith_zs_": "pot_zs",
    }
    method = "unknown"
    prefix = stem
    for candidate, parsed_method in prefixes.items():
        if stem.startswith(candidate):
            method = parsed_method
            prefix = candidate
            break
    rest = stem[len(prefix) :]
    match = re.match(r"(?P<model>.+)_(?P<start>\d+)_(?P<end>\d+)$", rest)
    if not match:
        return dataset, method, rest, None, None
    return dataset, method, match.group("model"), int(match.group("start")), int(match.group("end"))


def _write_summary(runs: list[dict], path: Path) -> None:
    fields = ["file", "dataset", "method_label", "model", "n", "correct", "accuracy", "exec_errors", "missing", "paper"]
    with path.open("w", encoding="utf-8") as handle:
        handle.write(",".join(fields) + "\n")
        for run in runs:
            handle.write(",".join("" if run[f] is None else str(run[f]) for f in fields) + "\n")


def _plot_best_method_comparison(runs: list[dict], path: Path) -> None:
    best = {}
    for run in runs:
        key = (run["dataset"], run["method"])
        current = best.get(key)
        if current is None or (run["n"], run["accuracy"]) > (current["n"], current["accuracy"]):
            best[key] = run
    ordered = []
    for dataset in ["gsm8k", "aqua", "multiarith"]:
        for method in METHOD_ORDER:
            if (dataset, method) in best:
                ordered.append(best[(dataset, method)])
    groups = []
    for run in ordered:
        values = [(run["accuracy"], run["method"], "Project")]
        if run["paper"] is not None:
            values.append((run["paper"], "paper", "Paper"))
        groups.append((f"{run['dataset'].upper()} {run['method_label']}", values))
    _bar_chart(path, "Project Results vs Paper Baselines", groups, "Accuracy (%)", 105)


def _bar_chart(path: Path, title: str, groups: list[tuple[str, list[tuple[float, str, str]]]], ylabel: str, ymax: float, stacked: bool = False) -> None:
    width = max(900, len(groups) * 115)
    height = 540
    left, top, chart_w, chart_h = 80, 55, width - 130, 360
    group_w = chart_w / max(len(groups), 1)
    bar_w = min(32, group_w / 4)
    parts = [_svg_open(width, height), _text(width / 2, 28, title, 18, "middle", bold=True)]
    parts.extend(_axes(left, top, chart_w, chart_h, ymax, ylabel))
    for i, (label, values) in enumerate(groups):
        center = left + group_w * i + group_w / 2
        if stacked:
            base_y = top + chart_h
            total = 0.0
            for value, method, legend in values:
                h = chart_h * value / ymax
                base_y -= h
                parts.append(_rect(center - bar_w / 2, base_y, bar_w, h, COLORS.get(method, "#888")))
                total += value
            if total > 0:
                parts.append(_text(center, base_y - 5, f"{total:.1f}", 10, "middle"))
        else:
            offset = -(len(values) - 1) * bar_w * 0.65
            for j, (value, method, legend) in enumerate(values):
                x = center + offset + j * bar_w * 1.3
                h = chart_h * value / ymax
                y = top + chart_h - h
                parts.append(_rect(x - bar_w / 2, y, bar_w, h, COLORS.get(method, "#888")))
                parts.append(_text(x, y - 5, f"{value:.1f}", 10, "middle"))
        for n, line in enumerate(_wrap_label(label, 18)):
            parts.append(_text(center, top + chart_h + 22 + n * 13, line, 10, "middle"))
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def _line_chart(runs: list[dict], path: Path) -> None:
    selected = [run for run in runs if run["dataset"] == "gsm8k" and run["n"] >= 10]
    selected = sorted(selected, key=lambda run: (-run["n"], run["method"], run["file"]))[:6]
    width, height = 900, 540
    left, top, chart_w, chart_h = 80, 55, 670, 360
    parts = [_svg_open(width, height), _text(width / 2, 28, "Cumulative Accuracy Stability", 18, "middle", bold=True)]
    parts.extend(_axes(left, top, chart_w, chart_h, 105, "Accuracy (%)"))
    for run_i, run in enumerate(selected):
        correct = 0
        points = []
        for i, row in enumerate(run["rows"], start=1):
            correct += compute_accuracy([row], mode="exact")["correct"]
            x = left + (i - 1) / max(run["n"] - 1, 1) * chart_w
            y = top + chart_h - (correct / i * 100) / 105 * chart_h
            points.append(f"{x:.1f},{y:.1f}")
        color = COLORS.get(run["method"], "#888")
        parts.append(f'<polyline points="{" ".join(points)}" fill="none" stroke="{color}" stroke-width="2.5"/>')
        parts.append(_rect(770, 70 + run_i * 25, 14, 14, color))
        parts.append(_text(790, 82 + run_i * 25, _short_label(run).replace("\n", " "), 10, "start"))
    parts.append(_text(left + chart_w / 2, top + chart_h + 55, "Example index within run", 12, "middle"))
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def _axes(left: float, top: float, width: float, height: float, ymax: float, ylabel: str) -> list[str]:
    parts = []
    for tick in range(0, int(ymax) + 1, 20):
        y = top + height - tick / ymax * height
        parts.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left+width}" y2="{y:.1f}" stroke="#ddd"/>')
        parts.append(_text(left - 10, y + 4, str(tick), 10, "end"))
    parts.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top+height}" stroke="#333"/>')
    parts.append(f'<line x1="{left}" y1="{top+height}" x2="{left+width}" y2="{top+height}" stroke="#333"/>')
    parts.append(_text(18, top + height / 2, ylabel, 12, "middle", rotate=-90))
    return parts


def _short_label(run: dict) -> str:
    return f"{run['method_label']}\n{run['model']}\nn={run['n']}"


def _wrap_label(label: str, max_len: int) -> list[str]:
    lines = []
    for part in label.split("\n"):
        if len(part) <= max_len:
            lines.append(part)
        else:
            chunks = part.split()
            line = ""
            for chunk in chunks:
                if len(line) + len(chunk) + 1 > max_len:
                    lines.append(line)
                    line = chunk
                else:
                    line = f"{line} {chunk}".strip()
            if line:
                lines.append(line)
    return lines[:4]


def _svg_open(width: int, height: int) -> str:
    return f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}"><rect width="100%" height="100%" fill="white"/>'


def _rect(x: float, y: float, width: float, height: float, fill: str) -> str:
    return f'<rect x="{x:.1f}" y="{y:.1f}" width="{width:.1f}" height="{height:.1f}" fill="{fill}"/>'


def _text(x: float, y: float, value: str, size: int, anchor: str, bold: bool = False, rotate: int | None = None) -> str:
    transform = f' transform="rotate({rotate} {x} {y})"' if rotate else ""
    weight = "700" if bold else "400"
    return f'<text x="{x:.1f}" y="{y:.1f}" text-anchor="{anchor}" font-family="Arial, sans-serif" font-size="{size}" font-weight="{weight}" fill="#222"{transform}>{html.escape(str(value))}</text>'


if __name__ == "__main__":
    main()
