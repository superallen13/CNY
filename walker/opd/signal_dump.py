from __future__ import annotations

import csv
import io
import sys

import torch

from walker.opd.per_step_teacher import ActionSpan

_PREFIX = "[OPD_CSV] "
_HEADER = [
    "step",
    "sample_idx",
    "span_idx",
    "pos_in_response",
    "token_id",
    "role",
    "student_lp",
    "teacher_lp",
    "abs_delta",
]
_HEADER_PRINTED = False

def _emit(row: list) -> None:
    
    buf = io.StringIO()
    csv.writer(buf).writerow(row)
    sys.stdout.write(_PREFIX + buf.getvalue())
    sys.stdout.flush()

def dump_opd_signal_rows(
    *,
    step_idx: int,
    sample_idx: int,
    spans: list[ActionSpan],
    student_log_probs: torch.Tensor,
    teacher_log_probs: torch.Tensor,
    response_token_ids: list[int],
    open_tag_tokens: int,
    close_tag_tokens: int,
) -> None:
    
    global _HEADER_PRINTED
    if not _HEADER_PRINTED:
        _HEADER_PRINTED = True
        _emit(_HEADER)

    for span_idx, span in enumerate(spans):
        x_start = span.start + open_tag_tokens
        x_end = span.end - close_tag_tokens
        for pos in range(span.start, span.end):
            if pos < x_start:
                role = "open_tag"
            elif pos < x_end:
                role = "x"
            else:
                role = "close_tag"
            student_lp = float(student_log_probs[pos].item())
            teacher_lp = float(teacher_log_probs[pos].item())
            _emit(
                [
                    step_idx,
                    sample_idx,
                    span_idx,
                    pos,
                    int(response_token_ids[pos]),
                    role,
                    f"{student_lp:.6f}",
                    f"{teacher_lp:.6f}",
                    f"{abs(teacher_lp - student_lp):.6f}",
                ]
            )
