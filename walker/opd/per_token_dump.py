from __future__ import annotations

import csv
import io
import json
import sys

import torch

from walker.opd.per_step_teacher import ActionSpan

_PREFIX = "[OPD_FULL] "
_HINT_PREFIX = "[OPD_FULL_HINT] "
_PROMPT_PREFIX = "[OPD_FULL_PROMPT] "
_REWARD_PREFIX = "[OPD_FULL_REWARD] "
_HEADER = [
    "cell_id",
    "step",
    "sample_idx",
    "pos",
    "token_id",
    "role",
    "loss_mask",
    "opd_action_mask",
    "student_lp",
    "teacher_lp",
    "outcome_reward",
]
_HEADER_PRINTED = False

def _emit(row: list) -> None:
    buf = io.StringIO()
    csv.writer(buf).writerow(row)
    sys.stdout.write(_PREFIX + buf.getvalue())
    sys.stdout.flush()

def _emit_hint(payload: dict) -> None:
    sys.stdout.write(_HINT_PREFIX + json.dumps(payload, ensure_ascii=False) + "\n")
    sys.stdout.flush()

def _emit_prompt(payload: dict) -> None:
    sys.stdout.write(_PROMPT_PREFIX + json.dumps(payload, ensure_ascii=False) + "\n")
    sys.stdout.flush()

def emit_reward(*, cell_id: str, step_idx: int, sample_idx: int, reward: float, correct: bool | None, metadata: dict | None) -> None:
    
    md = metadata or {}
    sys.stdout.write(_REWARD_PREFIX + json.dumps({
        "cell_id": cell_id,
        "step": step_idx,
        "sample_idx": sample_idx,
        "reward": float(reward) if reward is not None else None,
        "correct": bool(correct) if correct is not None else None,
        "node_reward_base": md.get("node_reward_base"),
        "node_reward_accuracy": md.get("node_reward_accuracy"),
        "node_valid_format": md.get("node_valid_format"),
        "node_has_answer": md.get("node_has_answer"),
        "node_search_count": md.get("node_search_count"),
    }, ensure_ascii=False, default=str) + "\n")
    sys.stdout.flush()

def _classify_position(
    pos: int,
    spans: list[ActionSpan],
    open_tag_tokens: int,
    close_tag_tokens: int,
) -> str:
    for span in spans:
        x_start = span.start + open_tag_tokens
        x_end = span.end - close_tag_tokens
        if span.start <= pos < x_start:
            return "tag"
        if x_start <= pos < x_end:
            return "x"
        if x_end <= pos < span.end:
            return "tag"
    return "outside"

def dump_per_token_rows(
    *,
    cell_id: str,
    step_idx: int,
    sample_idx: int,
    spans: list[ActionSpan],
    student_log_probs: torch.Tensor,
    teacher_log_probs: torch.Tensor,
    response_token_ids: list[int],
    loss_mask: list[int],
    opd_action_mask: list[int],
    outcome_reward: float,
    open_tag_tokens: int,
    close_tag_tokens: int,
    hints: list[str] | None = None,
    prompt_text: str | None = None,
) -> None:
    
    global _HEADER_PRINTED
    if not _HEADER_PRINTED:
        _HEADER_PRINTED = True
        _emit(_HEADER)

    if prompt_text is not None:
        _emit_prompt({
            "cell_id": cell_id,
            "step": step_idx,
            "sample_idx": sample_idx,
            "prompt_text": prompt_text,
        })

    response_length = len(response_token_ids)
    assert (
        student_log_probs.shape[0]
        == teacher_log_probs.shape[0]
        == len(loss_mask)
        == len(opd_action_mask)
        == response_length
    ), (
        f"per_token_dump length mismatch: "
        f"response_token_ids={response_length}, "
        f"student_lp={student_log_probs.shape[0]}, "
        f"teacher_lp={teacher_log_probs.shape[0]}, "
        f"loss_mask={len(loss_mask)}, "
        f"opd_action_mask={len(opd_action_mask)}"
    )

    if hints is not None:
        assert len(hints) == len(spans), (
            f"per_token_dump hints length {len(hints)} != spans length {len(spans)}"
        )
        for span_idx, (span, hint) in enumerate(zip(spans, hints)):
            _emit_hint({
                "cell_id": cell_id,
                "step": step_idx,
                "sample_idx": sample_idx,
                "span_idx": span_idx,
                "first_x_pos": span.start + open_tag_tokens,
                "hint_text": hint,
            })

    for pos in range(response_length):
        role = _classify_position(pos, spans, open_tag_tokens, close_tag_tokens)
        student_lp = float(student_log_probs[pos].item())
        teacher_lp = float(teacher_log_probs[pos].item())
        _emit([
            cell_id,
            step_idx,
            sample_idx,
            pos,
            int(response_token_ids[pos]),
            role,
            int(loss_mask[pos]),
            int(opd_action_mask[pos]),
            f"{student_lp:.6f}",
            f"{teacher_lp:.6f}",
            f"{outcome_reward:.6f}",
        ])
