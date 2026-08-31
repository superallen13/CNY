from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"


def fetch_hint(missing: list[Path]) -> str:
    rel = "\n".join(f"  {p.relative_to(REPO_ROOT)}" for p in missing)
    repo = os.environ.get("CNY_DATA_REPO", "") or "$CNY_DATA_REPO"
    return (
        f"missing data files:\n{rel}\n\n"
        f"prompts and graphs are distributed separately; fetch them with\n"
        f"  bash scripts/fetch_data.sh\n"
        f"(or: hf download {repo} --repo-type dataset --local-dir data)"
    )


def train_jsonl() -> Path:
    return DATA_DIR / "train.jsonl"


def eval_jsonl(dataset_name: str) -> Path:
    return DATA_DIR / "eval" / f"{dataset_name}.jsonl"


def graph_pt(dataset_name: str) -> Path:
    return DATA_DIR / "raw_datasets" / f"{dataset_name}.pt"
