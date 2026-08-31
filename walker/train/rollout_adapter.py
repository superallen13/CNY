from __future__ import annotations

import os

from slime.utils.types import Sample

from walker.search import rollout as _search_rollout

def _ensure_walk_extend_default() -> None:
    
    os.environ.setdefault("WALKER_WALK_EXTEND", "1")

_ensure_walk_extend_default()

async def generate(args, sample: Sample, sampling_params) -> Sample:
    
    sample = await _search_rollout.generate(args, sample, sampling_params)
    if "walk_path" not in sample.metadata:
        sample.metadata["walk_path"] = []
    return sample

log_rollout_data = _search_rollout.log_rollout_data
log_eval_rollout_data = _search_rollout.log_eval_rollout_data
post_process_rewards = _search_rollout.post_process_rewards

__all__ = [
    "generate",
    "log_rollout_data",
    "log_eval_rollout_data",
    "post_process_rewards",
]
