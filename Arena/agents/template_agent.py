from __future__ import annotations

from typing import Any

from .base import ArenaAgent


class TemplateAgent(ArenaAgent):
    """Example adapter skeleton for future agents such as X_Jiang."""

    def __init__(self, checkpoint: str | None = None, device: str = "cpu", **_: Any) -> None:
        self.checkpoint = checkpoint
        self.device = device

    def reset(self) -> None:
        pass

    def act(self, raw_observation: dict[str, Any], *, deterministic: bool = False) -> int:
        del raw_observation, deterministic
        raise NotImplementedError("Implement raw-observation preprocessing, model forward, and action mapping.")
