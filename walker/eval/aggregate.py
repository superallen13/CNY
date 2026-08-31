from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

CANON_COLUMNS: list[tuple[str, str]] = [
    ("cora", "Cora-Node"),
    ("wikics", "WikiCS"),
    ("products", "Products"),
    ("arxiv", "ArXiv"),
    ("cora_link", "Cora-Link"),
    ("fb15k237_edge_tag", "FB15K237"),
    ("expla_graph_tag", "Expla-Graph"),
]

_CANON_KEYS: list[str] = [k for k, _ in CANON_COLUMNS]
_CANON_DISPLAY: list[str] = [d for _, d in CANON_COLUMNS]

def row_key(meta: dict) -> str:
    
    return str(meta.get("row_label") or meta.get("model", ""))

def load_summaries(root: Path) -> list[dict]:
    
    results: list[dict] = []
    for p in sorted(root.rglob("*.summary.json")):
        try:
            results.append(json.loads(p.read_text()))
        except json.JSONDecodeError as exc:
            print(f"[aggregate] warning: skipping malformed {p}: {exc}", file=sys.stderr)
    return results

def build_cells(
    summaries: list[dict],
    *,
    value: str = "accuracy",
) -> dict[tuple[str, str], float]:
    
    cells: dict[tuple[str, str], float] = {}
    
    _chosen_n: dict[tuple[str, str], int] = {}

    for item in summaries:
        summary: dict[str, Any] = item.get("summary", {})
        meta: dict[str, Any] = item.get("meta", {})
        raw = summary.get(value)
        if raw is None:
            continue
        row_label = row_key(meta)
        dataset_name = str(meta.get("dataset", ""))
        key = (row_label, dataset_name)
        n = int(summary.get("n") or 0)
        existing_n = _chosen_n.get(key, -1)
        if n >= existing_n:  
            cells[key] = float(raw) * 100.0
            _chosen_n[key] = n

    return cells

def _build_row_data(
    summaries: list[dict],
    value: str,
) -> tuple[list[str], dict[str, dict[str, float]]]:
    
    cells = build_cells(summaries, value=value)
    row_labels: list[str] = sorted({k[0] for k in cells})
    row_data: dict[str, dict[str, float]] = {}
    for rl in row_labels:
        row_data[rl] = {ds: cells[(rl, ds)] for ds in _CANON_KEYS if (rl, ds) in cells}
    return row_labels, row_data

def _avg_str(present_vals: list[float]) -> str:
    if not present_vals:
        return "-"
    return f"{sum(present_vals) / len(present_vals):.1f}"

def render_markdown(summaries: list[dict], *, value: str = "accuracy") -> str:
    
    row_labels, row_data = _build_row_data(summaries, value)

    header = "| Method | " + " | ".join(_CANON_DISPLAY) + " | Avg |"
    sep = "| --- | " + " | ".join(["---"] * len(_CANON_DISPLAY)) + " | --- |"
    lines = [header, sep]

    for rl in row_labels:
        data = row_data[rl]
        cells_str: list[str] = []
        present: list[float] = []
        for ds in _CANON_KEYS:
            if ds in data:
                v = data[ds]
                cells_str.append(f"{v:.1f}")
                present.append(v)
            else:
                cells_str.append("-")
        avg = _avg_str(present)
        lines.append("| " + rl + " | " + " | ".join(cells_str) + " | " + avg + " |")

    return "\n".join(lines)

def render_dual_markdown(summaries: list[dict]) -> str:
    
    _, row_data_acc = _build_row_data(summaries, "accuracy")
    _, row_data_f1 = _build_row_data(summaries, "macro_f1")
    row_labels: list[str] = sorted(
        set(row_data_acc.keys()) | set(row_data_f1.keys())
    )

    header = "| Method | " + " | ".join(_CANON_DISPLAY) + " | Avg |"
    sep = "| --- | " + " | ".join(["---"] * len(_CANON_DISPLAY)) + " | --- |"
    lines = [header, sep]

    for rl in row_labels:
        acc_data = row_data_acc.get(rl, {})
        f1_data = row_data_f1.get(rl, {})
        cells_str: list[str] = []
        present_acc: list[float] = []
        present_f1: list[float] = []
        for ds in _CANON_KEYS:
            has_acc = ds in acc_data
            has_f1 = ds in f1_data
            if has_acc and has_f1:
                a, f = acc_data[ds], f1_data[ds]
                cells_str.append(f"{a:.1f}/{f:.1f}")
                present_acc.append(a)
                present_f1.append(f)
            elif has_acc:
                a = acc_data[ds]
                cells_str.append(f"{a:.1f}/-")
                present_acc.append(a)
            elif has_f1:
                f = f1_data[ds]
                cells_str.append(f"-/{f:.1f}")
                present_f1.append(f)
            else:
                cells_str.append("-")

        avg_acc = _avg_str(present_acc)
        avg_f1 = _avg_str(present_f1)
        if avg_acc == "-" and avg_f1 == "-":
            avg = "-"
        else:
            avg = f"{avg_acc}/{avg_f1}"
        lines.append("| " + rl + " | " + " | ".join(cells_str) + " | " + avg + " |")

    return "\n".join(lines)

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Aggregate walker benchmark *.summary.json files into a results table.",
    )
    ap.add_argument("--root", default="outputs/bench",
                    help="Directory to search recursively for *.summary.json files.")
    ap.add_argument("--value", choices=["accuracy", "macro_f1", "dual"], default="dual",
                    help="Metric to display. 'dual' shows acc/F1 combined (default).")
    args = ap.parse_args(argv)

    root = Path(args.root)
    summaries = load_summaries(root)
    if not summaries:
        print(f"[aggregate] no *.summary.json files found under {root}", file=sys.stderr)
        return 1

    if args.value == "dual":
        print(render_dual_markdown(summaries))
    else:
        print(render_markdown(summaries, value=args.value))
    return 0

if __name__ == "__main__":
    sys.exit(main())
