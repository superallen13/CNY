from __future__ import annotations

import asyncio

from walker.opd._render import render_template
from walker.opd.ego_graph import EgoGraph

async def _http_post(url: str, payload: dict) -> dict:
    from slime.utils.http_utils import post as _slime_post  

    return await _slime_post(url, payload)

_MAX_HINT_TOKENS = 64

class HintJudge:

    def __init__(
        self,
        *,
        server_url: str,
        max_new_tokens: int = _MAX_HINT_TOKENS,
        temperature: float = 0.0,
        template_name: str = "hint_judge",
    ) -> None:
        self.server_url = server_url
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.template_name = template_name
        self.stats: dict = {
            "call_count": 0,
            "fallback_count": 0,
            "hint_length_chars_sum": 0,
            "hint_length_chars_max": 0,
        }

    async def get_hint(self, ego: EgoGraph, *, target_label_name: str | None = None) -> str:
        
        self.stats["call_count"] += 1

        prompt_text = render_template(
            f"{self.template_name}.j2", ego=ego, target_label_name=target_label_name
        )

        if self.template_name.startswith("passthrough") or self.template_name == "hint_passthrough":
            n = len(prompt_text)
            self.stats["hint_length_chars_sum"] += n
            if n > self.stats["hint_length_chars_max"]:
                self.stats["hint_length_chars_max"] = n
            return prompt_text

        payload = {
            "text": prompt_text,
            "sampling_params": {
                "temperature": self.temperature,
                "max_new_tokens": self.max_new_tokens,

                "stop": ["<|im_end|>", "\n\n"],
                "skip_special_tokens": True,
            },
        }

        try:
            resp = await _http_post(self.server_url, payload)
            hint = (resp.get("text") or "").strip()
        except (asyncio.TimeoutError, Exception):  
            hint = ""

        if not hint:
            self.stats["fallback_count"] += 1
            hint = _fallback_hint(ego)

        n = len(hint)
        self.stats["hint_length_chars_sum"] += n
        if n > self.stats["hint_length_chars_max"]:
            self.stats["hint_length_chars_max"] = n
        return hint

def _fallback_hint(ego: EgoGraph) -> str:
    
    content_preview = (ego.content or "").strip().split("\n", 1)[0][:120]
    if content_preview:
        return f"Destination topic: {content_preview}"
    return "Destination has no extractable topic."
