from __future__ import annotations

from typing import Any

import numpy as np

from .base import ArenaAgent


class RandomAgent(ArenaAgent):
    def __init__(self, num_actions: int, seed: int | None = None, **_: Any) -> None:
        self._num_actions = num_actions
        self._rng = np.random.default_rng(seed)

    def act(self, raw_observation: dict[str, Any], *, deterministic: bool = False) -> int:
        del raw_observation, deterministic
        return int(self._rng.integers(self._num_actions))
