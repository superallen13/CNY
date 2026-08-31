from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Iterator

def load_label_index(label_emb_dir: str | Path, dataset: str) -> dict[str, int]:
    
    meta = json.loads((Path(label_emb_dir) / f"{dataset}.json").read_text())
    return {name: i for i, name in enumerate(meta["label_names"])}

def resolve_mask_ids(
    mask_labels: Iterable[str], label_index: dict[str, int]
) -> set[int]:
    
    out: set[int] = set()
    for name in mask_labels:
        if name not in label_index:
            valid = ", ".join(repr(n) for n in sorted(label_index))
            raise ValueError(
                f"unknown eval_mask_labels entry {name!r}; "
                f"valid label names: [{valid}]"
            )
        out.add(label_index[name])
    return out

def _row_label_id(row: dict) -> int:
    
    sol = row["solution"]
    if isinstance(sol, str):
        sol = json.loads(sol)
    return int(sol["ground_truth"])

def iter_jsonl_with_mask(
    jsonl_path: str | Path, mask_ids: set[int]
) -> Iterator[tuple[dict, bool]]:
    
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            yield row, _row_label_id(row) in mask_ids

def iter_eval_jsonl(
    jsonl_path: str | Path, mask_ids: set[int]
) -> Iterator[dict]:
    
    for row, excluded in iter_jsonl_with_mask(jsonl_path, mask_ids):
        if not excluded:
            yield row

def count_jsonl(jsonl_path: str | Path, mask_ids: set[int]) -> dict[str, int]:
    
    n_total = n_excluded = 0
    for _, excluded in iter_jsonl_with_mask(jsonl_path, mask_ids):
        n_total += 1
        if excluded:
            n_excluded += 1
    return {
        "total": n_total,
        "kept": n_total - n_excluded,
        "excluded": n_excluded,
    }
