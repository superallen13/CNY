from __future__ import annotations

import asyncio
from typing import Any

class Summarizer:

    def __init__(
        self,
        *,
        client: Any,
        model: str,
        template: str,
        max_tokens: int,
        temperature: float,
        enabled: bool,
    ) -> None:
        self.client = client
        self.model = str(model)
        self.template = str(template)
        self.max_tokens = int(max_tokens)
        self.temperature = float(temperature)
        self.enabled = bool(enabled)

    @classmethod
    def from_env_cfg(cls, env_cfg, *, client: Any, model: str) -> "Summarizer":
        
        return cls(
            client=client,
            model=str(model),
            template=str(env_cfg.summary_template),
            max_tokens=int(env_cfg.summary_max_tokens),
            temperature=float(env_cfg.summary_temperature),
            enabled=bool(env_cfg.summary_enabled),
        )

    async def summarize_batch(self, texts: list[str]) -> list[str]:
        
        if not self.enabled:
            return list(texts)
        if not texts:
            return []

        coros = [self._one(t) for t in texts]
        return await asyncio.gather(*coros)

    async def _one(self, text: str) -> str:
        resp = await self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": self.template.format(text=text)}],
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )
        return str(resp.choices[0].message.content or "").strip()
