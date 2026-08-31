from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

async def _http_post(url: str, payload: dict) -> dict:
    from slime.utils.http_utils import post as _slime_post
    return await _slime_post(url, payload)

_OPEN_TAG = "<walk>"
_CLOSE_TAG = "</walk>"

def _extract_logprob(item: Any) -> float | None:
    
    if item is None:
        return None
    if isinstance(item, dict):
        lp = item.get("logprob")
    else:
        lp = item[0] if len(item) > 0 else None
    return None if lp is None else float(lp)

@dataclass(frozen=True)
class ActionSpan:

    start: int
    end: int

class PerStepTeacherClient:

    def __init__(self, *, tokenizer: Any, server_url: str) -> None:
        self.tokenizer = tokenizer
        self.server_url = server_url
        self._cached_tag_lens: tuple[int, int] | None = None

    def _tag_token_lengths(self) -> tuple[int, int]:
        if self._cached_tag_lens is None:
            o = self.tokenizer.encode(_OPEN_TAG, add_special_tokens=False)
            c = self.tokenizer.encode(_CLOSE_TAG, add_special_tokens=False)
            self._cached_tag_lens = (len(o), len(c))
        return self._cached_tag_lens

    def _build_payload(
        self,
        *,
        prompt_text: str,
        response_prefix_ids: list[int],
        hint_text: str,
        action_token_ids: list[int],
    ) -> tuple[dict, int]:
        
        prompt_ids = self.tokenizer.encode(prompt_text, add_special_tokens=False)
        hint_block_text = f"\n[Hint about destination: {hint_text}]\n"
        hint_block_ids = self.tokenizer.encode(hint_block_text, add_special_tokens=False)
        full_ids = (
            list(prompt_ids)
            + list(response_prefix_ids)
            + list(hint_block_ids)
            + list(action_token_ids)
        )
        action_token_len = len(action_token_ids)

        start_len = max(0, len(full_ids) - action_token_len - 1)

        payload = {
            "input_ids": full_ids,
            "sampling_params": {
                "temperature": 0.0,
                "max_new_tokens": 0,
                "skip_special_tokens": False,
            },
            "return_logprob": True,
            "logprob_start_len": start_len,
        }
        return payload, action_token_len

    async def call_one_step(
        self,
        *,
        prompt_text: str,
        response_prefix_ids: list[int],
        hint_text: str,
        action_token_ids: list[int],
    ) -> torch.Tensor:
        
        payload, action_token_len = self._build_payload(
            prompt_text=prompt_text,
            response_prefix_ids=response_prefix_ids,
            hint_text=hint_text,
            action_token_ids=action_token_ids,
        )
        resp = await _http_post(self.server_url, payload)
        token_lps = resp.get("meta_info", {}).get("input_token_logprobs", [])

        lps = [_extract_logprob(item) for item in token_lps]
        lps = [lp for lp in lps if lp is not None]
        if len(lps) < action_token_len:
            raise RuntimeError(
                f"per-step teacher returned {len(lps)} logprobs, expected at "
                f"least {action_token_len} (action_token_ids has "
                f"{len(action_token_ids)} tokens; start_len was anchored one "
                f"position before the action span — see _build_payload)"
            )
        lps = lps[-action_token_len:]
        return torch.tensor(lps, dtype=torch.float32)

def assemble_teacher_log_probs(
    *,
    response_length: int,
    student_log_probs: torch.Tensor,
    action_spans: list[ActionSpan],
    per_step_teacher_logprobs: list[torch.Tensor],
    open_tag_tokens: int,
    close_tag_tokens: int,
) -> torch.Tensor:
    
    if student_log_probs.shape[0] != response_length:
        raise ValueError(
            f"student_log_probs length {student_log_probs.shape[0]} "
            f"!= response_length {response_length}"
        )
    out = student_log_probs.clone()
    target_device = out.device
    target_dtype = out.dtype
    for span, lps in zip(action_spans, per_step_teacher_logprobs):
        span_len = span.end - span.start
        if lps.shape[0] != span_len:
            raise ValueError(
                f"span length {span_len} != lps length {lps.shape[0]} "
                f"at span {span}"
            )
        if open_tag_tokens + close_tag_tokens > span_len:
            raise ValueError(
                f"open+close ({open_tag_tokens}+{close_tag_tokens}) > "
                f"span length {span_len}"
            )
        x_start = span.start + open_tag_tokens
        x_end = span.end - close_tag_tokens
        
        slice_lps = lps[open_tag_tokens : span_len - close_tag_tokens]
        out[x_start:x_end] = slice_lps.to(device=target_device, dtype=target_dtype)
        
    return out
