from __future__ import annotations

import os
from typing import Any

import torch

from walker.search._env_compat import get_env
from walker.opd.ego_graph import EgoGraph, NeighborPreview
from walker.opd.hint_judge import HintJudge
from walker.opd.per_step_teacher import (
    ActionSpan,
    PerStepTeacherClient,
    assemble_teacher_log_probs,
)

def build_opd_action_mask(
    *,
    response_length: int,
    action_spans: list[ActionSpan],
    open_tag_tokens: int,
    close_tag_tokens: int,
) -> torch.Tensor:
    
    mask = torch.zeros(response_length, dtype=torch.bool)
    for span in action_spans:
        x_start = span.start + open_tag_tokens
        x_end = span.end - close_tag_tokens
        if x_end > x_start:
            mask[x_start:x_end] = True
    return mask

def is_active(args: Any) -> bool:
    
    return getattr(args, "method", None) == "opd" and bool(getattr(args, "enable_opd", False))

def _compute_opd_diagnostics(
    *,
    student_log_probs: torch.Tensor,
    teacher_log_probs: torch.Tensor,
    action_spans: list[ActionSpan],
    open_tag_tokens: int,
    close_tag_tokens: int,
) -> dict[str, float]:
    
    reverse_kl = (student_log_probs.float() - teacher_log_probs.float())

    n = reverse_kl.numel()
    x_mask = torch.zeros(n, dtype=torch.bool)
    tag_mask = torch.zeros(n, dtype=torch.bool)
    for span in action_spans:
        x_start = span.start + open_tag_tokens
        x_end = span.end - close_tag_tokens
        if x_end > x_start:
            x_mask[x_start:x_end] = True
        
        tag_mask[span.start:span.start + open_tag_tokens] = True
        tag_mask[span.end - close_tag_tokens:span.end] = True

    n_x = int(x_mask.sum())
    n_tag = int(tag_mask.sum())

    out: dict[str, float] = {
        "x_token_count": float(n_x),
        "tag_token_count": float(n_tag),
        "x_token_fraction": (n_x / n) if n > 0 else 0.0,
    }

    if n_x > 0:
        rk_x = reverse_kl[x_mask]
        out["reverse_kl_signal_mean"] = rk_x.mean().item()
        out["reverse_kl_signal_max_abs"] = rk_x.abs().max().item()
        out["teacher_lp_on_action_mean"] = teacher_log_probs[x_mask].float().mean().item()
        if n_x > 1:
            out["reverse_kl_signal_std"] = rk_x.std().item()
            
            q = torch.tensor([0.01, 0.99], dtype=torch.float32)
            p1, p99 = torch.quantile(rk_x, q).tolist()
            out["reverse_kl_signal_p1"] = p1
            out["reverse_kl_signal_p99"] = p99

    if n_tag > 0:

        out["tag_reverse_kl_max_abs"] = reverse_kl[tag_mask].abs().max().item()

    return out

async def apply_opd_branch(
    args: Any,
    sample: Any,
    *,
    tokenizer: Any,
    prompt_text: str,
    response_token_ids: list[int],
    graph_state: Any,
    meta: dict,
) -> None:
    
    teacher_url = getattr(args, "opd_teacher_url", None)
    if not teacher_url:
        raise RuntimeError(
            "method=opd requires args.opd_teacher_url (frozen-teacher SGLang). "
            "Set --opd-teacher-url in the launcher."
        )

    reuse_student = teacher_url == "auto-router"
    if reuse_student:
        router_ip = getattr(args, "sglang_router_ip", None)
        router_port = getattr(args, "sglang_router_port", None)
        if not router_ip or not router_port:
            raise RuntimeError(
                "opd_teacher_url='auto-router' requires args.sglang_router_ip "
                "and args.sglang_router_port to be set by slime. The router "
                "should have started before per_step_teacher runs."
            )
        teacher_url = f"http://{router_ip}:{router_port}/generate"

    spans_raw: list[tuple[int, int]] = list(meta.get("search_action_spans") or [])
    walk_destinations: list[int] = list(meta.get("node_walk_path") or [])

    if len(spans_raw) != len(walk_destinations):

        raise RuntimeError(
            f"search_action_spans ({len(spans_raw)}) and node_walk_path "
            f"({len(walk_destinations)}) length mismatch — rollout step "
            f"tracking is broken."
        )

    student_lp_list = getattr(sample, "rollout_log_probs", None)
    if student_lp_list is None:
        raise RuntimeError(
            "method=opd requires sample.rollout_log_probs (i.e., "
            "WALKER_RETURN_LOGPROB=1 in the rollout config); got None. "
            "Without student logprobs, OPD KL fires with student_lp=0 which "
            "produces an arbitrary-magnitude gradient at every X token, not a "
            "zero contribution."
        )
    student_log_probs = torch.tensor(student_lp_list, dtype=torch.float32)

    if not spans_raw:

        sample.teacher_log_probs = student_log_probs
        sample.opd_action_mask = torch.zeros(len(response_token_ids), dtype=torch.bool)
        sample.metadata["opd_metrics"] = {
            "x_token_count": 0.0,
            "tag_token_count": 0.0,
            "x_token_fraction": 0.0,
        }
        sample.metadata["hint_metrics"] = {
            "call_count": 0,
            "fallback_count": 0,
            "hint_length_chars_mean": 0.0,
            "hint_length_chars_max": 0,
        }
        return

    if not reuse_student and os.environ.get("WALKER_TEACHER_SLEEP", "0") == "1":
        try:
            from walker.opd.teacher_sleep_state import claim_resume
            from walker.opd.per_step_teacher import _http_post

            if claim_resume(teacher_url):
                base = teacher_url.rsplit("/", 1)[0] if teacher_url.endswith("/generate") else teacher_url
                await _http_post(f"{base}/resume_memory_occupation", {"tags": ["weights"]})
        except Exception:  
            
            pass

    raw_label = getattr(sample, "label", None)
    gold_int: int | None = None
    if isinstance(raw_label, dict):
        g = raw_label.get("ground_truth")
        if isinstance(g, (int, str)) and str(g).lstrip("-").isdigit():
            gold_int = int(g)
    elif isinstance(raw_label, int):
        gold_int = raw_label
    target_label_name: str | None = None
    label_names = getattr(getattr(graph_state, "tag", None), "label_names", None)
    if gold_int is not None and label_names and 0 <= gold_int < len(label_names):
        target_label_name = label_names[gold_int]

    judge = HintJudge(
        server_url=teacher_url,
        template_name=os.environ.get("WALKER_HINT_TEMPLATE", "hint_judge"),
    )
    client = PerStepTeacherClient(tokenizer=tokenizer, server_url=teacher_url)
    open_tok_len, close_tok_len = client._tag_token_lengths()

    snapshots = list(meta.get("walk_ego_snapshots") or [])
    if len(snapshots) != len(spans_raw):
        raise RuntimeError(
            f"walk_ego_snapshots ({len(snapshots)}) and search_action_spans "
            f"({len(spans_raw)}) length mismatch — rollout-side wiring is broken."
        )

    per_step_lps: list[torch.Tensor] = []
    per_step_hints: list[str] = []
    for snap, dest_id, (s_start, s_end) in zip(snapshots, walk_destinations, spans_raw):
        assert snap.node_id == dest_id, (
            f"snapshot.node_id ({snap.node_id}) != walk dest ({dest_id}) — "
            f"alignment bug between rollout span tracker and ego capture."
        )
        ego = EgoGraph(
            node_id=snap.node_id,
            title=str(snap.node_id),
            content=snap.content,
            neighbors=tuple(
                NeighborPreview(id=int(nid), title=str(nid), preview=str(pv))
                for nid, pv in zip(snap.neighbor_ids, snap.neighbor_previews)
            ),
            label=snap.label,
            label_name=snap.label_name,
        )
        hint = await judge.get_hint(ego, target_label_name=target_label_name)
        per_step_hints.append(hint)

        action_token_ids = list(response_token_ids[s_start:s_end])

        response_prefix_ids = list(response_token_ids[:s_start])
        lps = await client.call_one_step(
            prompt_text=prompt_text,
            response_prefix_ids=response_prefix_ids,
            hint_text=hint,
            action_token_ids=action_token_ids,
        )
        per_step_lps.append(lps)

    action_spans = [ActionSpan(start=a, end=b) for (a, b) in spans_raw]
    sample.teacher_log_probs = assemble_teacher_log_probs(
        response_length=len(response_token_ids),
        student_log_probs=student_log_probs,
        action_spans=action_spans,
        per_step_teacher_logprobs=per_step_lps,
        open_tag_tokens=open_tok_len,
        close_tag_tokens=close_tok_len,
    )
    sample.opd_action_mask = build_opd_action_mask(
        response_length=len(response_token_ids),
        action_spans=action_spans,
        open_tag_tokens=open_tok_len,
        close_tag_tokens=close_tok_len,
    )

    sample.metadata["opd_metrics"] = _compute_opd_diagnostics(
        student_log_probs=student_log_probs,
        teacher_log_probs=sample.teacher_log_probs,
        action_spans=action_spans,
        open_tag_tokens=open_tok_len,
        close_tag_tokens=close_tok_len,
    )

    cc = int(judge.stats["call_count"])
    lp_sum = int(judge.stats["hint_length_chars_sum"])
    sample.metadata["hint_metrics"] = {
        "call_count": cc,
        "fallback_count": int(judge.stats["fallback_count"]),
        "hint_length_chars_mean": (lp_sum / cc) if cc > 0 else 0.0,
        "hint_length_chars_max": int(judge.stats["hint_length_chars_max"]),
    }

    if get_env("WALKER_DUMP_OPD_CSV", default="0") == "1":
        from walker.opd.signal_dump import dump_opd_signal_rows
        dump_opd_signal_rows(

            step_idx=int(get_env("WALKER_TRAINING_STEP", default="0") or "0"),
            sample_idx=int(sample.index if sample.index is not None else 0),
            spans=action_spans,
            student_log_probs=student_log_probs,
            teacher_log_probs=sample.teacher_log_probs,
            response_token_ids=response_token_ids,
            open_tag_tokens=open_tok_len,
            close_tag_tokens=close_tok_len,
        )
    if get_env("WALKER_DUMP_PER_TOKEN", default="0") == "1":
        from walker.opd.per_token_dump import dump_per_token_rows
        dump_per_token_rows(
            cell_id=get_env("WALKER_DUMP_CELL_ID", default="unknown"),
            step_idx=int(get_env("WALKER_TRAINING_STEP", default="0") or "0"),
            sample_idx=int(sample.index if sample.index is not None else 0),
            spans=action_spans,
            student_log_probs=student_log_probs,
            teacher_log_probs=sample.teacher_log_probs,
            response_token_ids=response_token_ids,
            loss_mask=list(sample.loss_mask) if sample.loss_mask is not None else [1] * len(response_token_ids),
            opd_action_mask=list(sample.opd_action_mask) if sample.opd_action_mask is not None else [0] * len(response_token_ids),
            outcome_reward=float(sample.reward if sample.reward is not None else 0.0),
            open_tag_tokens=open_tok_len,
            close_tag_tokens=close_tok_len,
            hints=per_step_hints,
            prompt_text=prompt_text,
        )
