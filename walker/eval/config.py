from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Optional

from walker._yaml_loader import apply_overrides, resolve_extends

@dataclass
class EvalConfig:
    
    eval_dataset: str = "cora"
    eval_subsample: Optional[int] = 200

    walker_force_min_search: int = 1
    walker_max_hops: int = 5
    walker_walk_extend: bool = True
    walker_neighbor_limit: int = 3
    walker_node_content_tokens: int = 120
    walker_preview_tokens: int = 20

    walker_summary_enabled: bool = False
    walker_summary_max_tokens: int = 40
    walker_summary_temperature: float = 0.0
    walker_summary_template: str = ""  

    backend_url: str = "http://127.0.0.1:30000/v1"
    model: str = "qwen3-4b"
    api_key_env: Optional[str] = None  

    max_in_flight: int = 32
    max_tokens: int = 512
    temperature: float = 0.0
    max_turns: int = 5  

    no_think: bool = False

    use_wandb: bool = False
    wandb_project: str = "graphwalker_bench"
    wandb_group: Optional[str] = None
    wandb_notes: Optional[str] = None
    wandb_tags: Optional[str] = None

def _build(raw: dict) -> EvalConfig:
    expected = {f.name for f in fields(EvalConfig)}
    extra = sorted(set(raw) - expected)
    if extra:
        raise ValueError(
            f"unknown keys in eval yaml: {extra}\n"
            f"valid keys are defined in walker.eval.config.EvalConfig"
        )
    return EvalConfig(**raw)

def load_eval(yaml_path: Path | str, overrides: list[str] | None = None) -> EvalConfig:
    
    raw = resolve_extends(Path(yaml_path))
    apply_overrides(raw, overrides)
    return _build(raw)
