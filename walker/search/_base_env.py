from __future__ import annotations

class BaseInteractionEnv:

    def reset(self):
        raise NotImplementedError

    def step(self, response_text: str):
        raise NotImplementedError

    def close(self):
        pass
