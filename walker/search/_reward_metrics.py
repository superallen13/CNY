from __future__ import annotations

import re
from typing import Any

from slime.rollout.sglang_rollout import GenerateState
from slime.utils.types import Sample

from walker.search._env_compat import get_env
from walker.search._format import (
    _evaluate_policy_output,
    _rebuild_policy_text,
    _score_prediction,
)
from walker.search.env import _normalize_label, _normalize_search_id

def _collect_reward_metrics(args: Any, samples: list[Sample], prefix: str) -> dict:
    
    if not samples:
        return {}

    tokenizer = GenerateState(args).tokenizer

    scores, accs = [], []
    valid_format_list, has_answer_list = [], []
    has_search_list, valid_search_list = [], []
    reward_base, reward_penalty = [], []
    direct_answer_rate = []
    search_count, invalid_search_attempt = [], []
    turns_used_list, budget_exhausted, answered_list = [], [], []
    response_length_list, truncated_action_list = [], []
    finish_stop, finish_length, finish_abort = [], [], []
    walk_hops_list, walk_extend_list = [], []
    force_min_triggers_list: list[float] = []
    force_min_any_list: list[float] = []
    non_canonical_list: list[float] = []
    non_canonical_any_list: list[float] = []
    acc_with_search_num, acc_with_search_den = 0.0, 0
    acc_no_search_num, acc_no_search_den = 0.0, 0

    for s in samples:
        scores.append(float(s.reward) if s.reward is not None else 0.0)

        label = _normalize_label(s.label)

        label_id    = int(label.get("ground_truth", -1))
        label_names = label.get("label_names", [])
        task_type   = str(label.get("task_type", "node_class"))
        target_ids  = {int(x) for x in label.get("neighbor_ids", [])}
        walk_ids    = {int(x) for x in (s.metadata or {}).get("node_walk_valid_ids", [])}
        valid_ids   = target_ids | walk_ids

        policy_text = _rebuild_policy_text(s, tokenizer)
        eval_info = _evaluate_policy_output(policy_text, valid_ids)
        correct = (
            eval_info["has_answer"]
            and _score_prediction(eval_info["answer_content"] or "", label_id, label_names, task_type)
        )

        accs.append(1.0 if correct else 0.0)
        valid_format_list.append(1.0 if eval_info["valid_format"] else 0.0)
        has_answer_list.append(1.0 if eval_info["has_answer"] else 0.0)
        has_search_list.append(1.0 if eval_info["has_search"] else 0.0)
        valid_search_list.append(1.0 if eval_info["valid_search"] else 0.0)
        direct_answer_rate.append(
            1.0 if (eval_info["has_answer"] and not eval_info["has_search"]) else 0.0
        )

        metadata = dict(s.metadata or {})
        reward_base.append(float(metadata.get("node_reward_base", 0.0) or 0.0))
        reward_penalty.append(float(metadata.get("node_reward_penalty", 0.0) or 0.0))

        all_searches = re.findall(r"<walk>\s*(.*?)\s*</walk>", policy_text, re.DOTALL)
        search_count.append(float(len(all_searches)))
        has_invalid = any(
            _normalize_search_id(raw.strip(), valid_ids) is None for raw in all_searches
        )
        invalid_search_attempt.append(1.0 if has_invalid else 0.0)

        turns_used_list.append(float(metadata.get("node_turns_used", 0) or 0))
        budget_exhausted.append(1.0 if metadata.get("node_budget_exhausted") else 0.0)
        answered_list.append(1.0 if metadata.get("node_answered") else 0.0)
        truncated_action_list.append(1.0 if metadata.get("node_truncated_action") else 0.0)
        response_length_list.append(float(getattr(s, "response_length", 0) or 0))

        reason = str(metadata.get("node_finish_reason", ""))
        finish_stop.append(1.0 if reason == "stop" else 0.0)
        finish_length.append(1.0 if reason == "length" else 0.0)
        finish_abort.append(1.0 if reason == "abort" else 0.0)

        walk_hops_val = int(metadata.get("node_walk_hops", 0) or 0)
        walk_hops_list.append(float(walk_hops_val))
        walk_extend_list.append(1.0 if walk_hops_val >= 2 else 0.0)

        fm_triggers = int(metadata.get("node_force_min_triggers", 0) or 0)
        force_min_triggers_list.append(float(fm_triggers))
        force_min_any_list.append(1.0 if fm_triggers > 0 else 0.0)

        nc_attempts = int(metadata.get("node_non_canonical_search_attempts", 0) or 0)
        non_canonical_list.append(float(nc_attempts))
        non_canonical_any_list.append(1.0 if nc_attempts > 0 else 0.0)

        if len(all_searches) > 0:
            acc_with_search_num += 1.0 if correct else 0.0
            acc_with_search_den += 1
        else:
            acc_no_search_num += 1.0 if correct else 0.0
            acc_no_search_den += 1

    def _mean(lst: list) -> float:
        return sum(lst) / len(lst) if lst else 0.0

    metrics = {
        f"{prefix}/reward_mean":                 _mean(scores),
        f"{prefix}/acc_rate":                    _mean(accs),
        f"{prefix}/valid_format_rate":           _mean(valid_format_list),
        f"{prefix}/has_answer_rate":             _mean(has_answer_list),
        f"{prefix}/has_search_rate":             _mean(has_search_list),
        f"{prefix}/valid_search_rate":           _mean(valid_search_list),
        f"{prefix}/direct_answer_rate":          _mean(direct_answer_rate),
        f"{prefix}/reward_base_mean":            _mean(reward_base),
        f"{prefix}/reward_penalty_mean":         _mean(reward_penalty),
        f"{prefix}/search_count_mean":           _mean(search_count),
        f"{prefix}/turns_used_mean":             _mean(turns_used_list),
        f"{prefix}/answered_rate":               _mean(answered_list),
        f"{prefix}/budget_exhausted_rate":       _mean(budget_exhausted),
        f"{prefix}/response_length_mean":        _mean(response_length_list),
        f"{prefix}/truncated_action_rate":       _mean(truncated_action_list),
        f"{prefix}/finish_stop_rate":            _mean(finish_stop),
        f"{prefix}/finish_length_rate":          _mean(finish_length),
        f"{prefix}/finish_abort_rate":           _mean(finish_abort),
        f"{prefix}/invalid_search_attempt_rate": _mean(invalid_search_attempt),
        f"{prefix}/walk_hops_mean":              _mean(walk_hops_list),
        f"{prefix}/walk_extend_rate":            _mean(walk_extend_list),
        f"{prefix}/force_min_triggers_mean":     _mean(force_min_triggers_list),
        f"{prefix}/force_min_trigger_rate":      _mean(force_min_any_list),
        f"{prefix}/non_canonical_search_mean":   _mean(non_canonical_list),
        f"{prefix}/non_canonical_search_rate":   _mean(non_canonical_any_list),
    }

    if acc_with_search_den > 0:
        metrics[f"{prefix}/acc_given_search"] = acc_with_search_num / acc_with_search_den
    if acc_no_search_den > 0:
        metrics[f"{prefix}/acc_given_no_search"] = acc_no_search_num / acc_no_search_den

    return metrics

def _collect_opd_metrics(samples: list[Sample], prefix: str) -> dict:
    
    if not samples:
        return {}

    accum: dict[str, list[float]] = {}
    for s in samples:
        m = (s.metadata or {}).get("opd_metrics") or {}
        for k, v in m.items():
            accum.setdefault(k, []).append(float(v))

    if not accum:
        return {}

    out: dict[str, float] = {}
    for k, vals in accum.items():
        if vals:
            out[f"{prefix}/{k}"] = sum(vals) / len(vals)

    n_with_signal = sum(1 for s in samples
                        if ((s.metadata or {}).get("opd_metrics") or {}).get("x_token_count", 0) > 0)
    out[f"{prefix}/samples_with_opd_signal"] = n_with_signal / len(samples)

    hint_call_counts: list[float] = []
    hint_fallback_counts: list[float] = []
    hint_any_fallback: list[float] = []
    hint_length_means: list[float] = []
    hint_length_maxes: list[float] = []
    for s in samples:
        h = (s.metadata or {}).get("hint_metrics") or {}
        if not h:
            continue
        cc = float(h.get("call_count", 0))
        fc = float(h.get("fallback_count", 0))
        hint_call_counts.append(cc)
        hint_fallback_counts.append(fc)
        hint_any_fallback.append(1.0 if fc > 0 else 0.0)
        if "hint_length_chars_mean" in h:
            hint_length_means.append(float(h["hint_length_chars_mean"]))
        if "hint_length_chars_max" in h:
            hint_length_maxes.append(float(h["hint_length_chars_max"]))

    def _hint_mean(lst: list[float]) -> float:
        return sum(lst) / len(lst) if lst else 0.0

    if hint_call_counts:
        out[f"{prefix}/hint_call_count_mean"] = _hint_mean(hint_call_counts)
        out[f"{prefix}/hint_fallback_count_mean"] = _hint_mean(hint_fallback_counts)
        out[f"{prefix}/hint_fallback_rate"] = _hint_mean(hint_any_fallback)
    if hint_length_means:
        out[f"{prefix}/hint_length_chars_mean"] = _hint_mean(hint_length_means)
    if hint_length_maxes:
        out[f"{prefix}/hint_length_chars_max"] = _hint_mean(hint_length_maxes)
    return out

def log_rollout_data(rollout_id, args, samples: list[Sample], extra_metrics, rollout_time) -> bool:
    
    if not samples or not getattr(args, "use_wandb", False):
        return False

    import wandb

    metrics = _collect_reward_metrics(args, samples, prefix="rollout")
    if metrics:
        wandb.log(metrics, step=rollout_id, commit=False)

    opd_metrics = _collect_opd_metrics(samples, prefix="opd")
    if opd_metrics:
        wandb.log(opd_metrics, step=rollout_id, commit=False)

    _maybe_log_sample_table(rollout_id, samples, prefix="rollout")

    return False

def _maybe_log_sample_table(rollout_id: int, samples: list[Sample], prefix: str) -> None:
    
    if get_env("WANDB_LOG_SAMPLE_TABLE", default="0") != "1":
        return
    interval = int(get_env("WANDB_SAMPLE_TABLE_INTERVAL", default="10"))
    if interval <= 0 or rollout_id % interval != 0:
        return
    max_rows = int(get_env("WANDB_SAMPLE_TABLE_MAX_ROWS", default="5"))
    if max_rows <= 0:
        return

    import random
    import wandb

    picks = random.sample(samples, k=min(max_rows, len(samples)))
    columns = ["rollout", "reward", "has_search", "has_answer", "valid_format",
               "response_len", "prompt_preview", "response"]
    table = wandb.Table(columns=columns)
    for s in picks:
        meta = s.metadata or {}
        prompt_text = s.prompt if isinstance(s.prompt, str) else str(s.prompt)
        reward = s.reward if isinstance(s.reward, (int, float)) else (
            (s.reward or {}).get("raw_reward") if isinstance(s.reward, dict) else None
        )
        table.add_data(
            int(rollout_id),
            float(reward) if reward is not None else None,
            bool(meta.get("node_has_search", False)),
            bool(meta.get("node_has_answer", False)),
            bool(meta.get("node_valid_format", False)),
            int(s.response_length or 0),
            prompt_text[-400:],
            (s.response or "")[:6000],
        )
    wandb.log({f"{prefix}/samples_at_{rollout_id}": table}, step=rollout_id, commit=False)

def log_eval_rollout_data(rollout_id, args, data: dict, extra_metrics) -> bool:
    
    if not data or not getattr(args, "use_wandb", False):
        return False

    import wandb

    per_dataset: list[dict] = []
    for dataset_name, info in data.items():
        samples = info.get("samples", [])
        if not samples:
            continue
        metrics = _collect_reward_metrics(args, samples, prefix=f"eval/{dataset_name}")
        if not metrics:
            continue
        wandb.log(metrics, step=rollout_id, commit=False)
        per_dataset.append({k.split("/", 2)[-1]: v for k, v in metrics.items()})

    HEADLINE = (
        "acc_rate",
        "reward_mean",
        "valid_format_rate",
        "has_answer_rate",
        "has_search_rate",
        "valid_search_rate",
        "search_count_mean",
        "walk_hops_mean",
        "response_length_mean",
    )
    if len(per_dataset) >= 2:
        overall: dict[str, float] = {}
        for key in HEADLINE:
            vals = [d[key] for d in per_dataset if key in d]
            if vals:
                overall[f"eval/overall/{key}"] = sum(vals) / len(vals)
        overall["eval/overall/num_datasets"] = float(len(per_dataset))
        if overall:
            wandb.log(overall, step=rollout_id, commit=False)

    return False
