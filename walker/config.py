from __future__ import annotations

from dataclasses import dataclass, field, fields, asdict
from pathlib import Path
from typing import Any, Optional

import yaml

from walker._yaml_loader import apply_overrides, resolve_extends

@dataclass
class ExpConfig:
    
    walker_hf_checkpoint: str = "Qwen/Qwen2.5-7B-Instruct"
    model_arch: str = "slime/scripts/models/qwen2.5-7B.sh"
    
    hf_local_dir_host: Optional[str] = None
    hf_local_dir_in_sif: Optional[str] = None
    hf_local_dir: str = "data/hf_ckpt/qwen2.5-7b"  
    
    override_rotary_base: Optional[int] = None

    train_datasets: list[str] = field(default_factory=lambda: [
        "citeseer", "pubmed", "photo", "computer",
        "history", "sportsfit", "instagram", "wn18rr_edge_tag",
    ])
    eval_datasets: list[str] = field(default_factory=lambda: ["cora", "wikics"])
    
    eval_subsample: Optional[int] = 200

    apply_chat_template_kwargs: dict = field(default_factory=dict)

    num_rollout: int = 50
    rollout_batch_size: int = 8
    n_samples_per_prompt: int = 4
    global_batch_size: int = 32
    rollout_max_response_len: int = 384
    rollout_temperature: float = 1.0
    eval_interval: int = 999999
    eval_before_train: bool = False  
    save_interval: int = 999999

    lr: float = 1e-6
    max_tokens_per_gpu: int = 2048
    log_probs_chunk_size: int = 512
    use_cpu_offload: bool = False
    precision_aware_adam: bool = False

    enable_opd: bool = True
    opd_kl_coef: float = 0.01

    use_rollout_logprobs: bool = False

    teacher_port: int = 13141
    teacher_gpu_index: str = "auto"
    teacher_mem_fraction: float = 0.6
    teacher_max_new_tokens: int = 64
    teacher_health_timeout_sec: int = 180
    walker_teacher_sleep: bool = False

    teacher_reuse_student: bool = False

    walker_prompt_template: Optional[str] = None
    walker_force_min_search: int = 1
    walker_max_hops: int = 5
    walker_node_content_tokens: int = 120
    walker_preview_tokens: int = 20
    walker_neighbor_limit: int = 3
    walker_return_logprob: bool = True
    walker_dump_opd_csv: bool = True
    walker_dump_per_token: bool = False
    walker_dump_cell_id: str = ""
    walker_walk_extend: bool = True
    walker_hint_template: str = "hint_judge"

    sglang_mem: float = 0.30
    sglang_context_length: int = 8192

    sglang_max_running_requests: int = 16
    sglang_disable_cuda_graph: bool = False
    sglang_chunked_prefill_size: Optional[int] = None
    sglang_attention_backend: Optional[str] = None
    sglang_deterministic: bool = False

    tensor_parallel_size: int = 1
    pipeline_parallel_size: int = 1

    cluster_num_nodes: Optional[int] = None
    cluster_gpus_per_node: int = 1
    rollout_num_gpus_per_engine: int = 1

    rollout_num_gpus: Optional[int] = None

    use_recompute: bool = True

    train_memory_margin_bytes: int = 0

    use_wandb: bool = True
    wandb_project: str = "cny"
    wandb_group: Optional[str] = None
    wandb_tags: Optional[str] = None
    wandb_notes: Optional[str] = None
    wandb_log_sample_table: bool = False
    wandb_sample_table_interval: int = 10
    wandb_sample_table_max_rows: int = 5

    debug_rollout_only: bool = False
    balance_to_dataset: Optional[str] = None

    sif_image: Optional[str] = None
    walker_venv: Optional[str] = None

def _build(raw: dict) -> ExpConfig:
    expected = {f.name for f in fields(ExpConfig)}
    extra = sorted(set(raw) - expected)
    if extra:
        raise ValueError(
            f"unknown keys in exp yaml: {extra}\n"
            f"valid keys are defined in walker.config.ExpConfig "
            f"(grep ExpConfig in walker/config.py)"
        )
    return ExpConfig(**raw)

def load_exp(yaml_path: Path | str, overrides: list[str] | None = None) -> ExpConfig:
    
    raw = resolve_extends(Path(yaml_path))
    apply_overrides(raw, overrides)
    return _build(raw)

def render_resolved(yaml_path: Path | str) -> str:
    
    yaml_path = Path(yaml_path)
    chain = _walk_chain(yaml_path)
    
    source: dict[str, Path] = {}
    merged: dict[str, Any] = {}
    for f in chain:
        raw = yaml.safe_load(f.read_text()) or {}
        raw.pop("extends", None)
        raw.pop("_doc", None)
        for k, v in raw.items():
            source[k] = f
            merged[k] = v

    cfg = load_exp(yaml_path)
    full = asdict(cfg)
    lines = [f"=== Effective config: {yaml_path} ==="]
    by_source: dict[str, list[tuple[str, Any]]] = {}
    for key, value in sorted(full.items()):
        src = source.get(key)
        bucket = str(src.relative_to(yaml_path.parent.parent)) if src else "(default)"
        by_source.setdefault(bucket, []).append((key, value))
    for bucket in sorted(by_source.keys()):
        lines.append(f"\n[from {bucket}]")
        for key, value in by_source[bucket]:
            lines.append(f"  {key:34s} = {value!r}")
    return "\n".join(lines)

def _walk_chain(path: Path) -> list[Path]:
    
    raw = yaml.safe_load(path.read_text()) or {}
    parent_name = raw.get("extends")
    if parent_name is None:
        return [path]
    if not (parent_name.endswith(".yaml") or parent_name.endswith(".yml")):
        parent_name = parent_name + ".yaml"
    parent_path = path.parent / parent_name
    return _walk_chain(parent_path) + [path]
