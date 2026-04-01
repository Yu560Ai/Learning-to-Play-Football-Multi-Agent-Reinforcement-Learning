from __future__ import annotations

from typing import Any

import numpy as np

from .base import ArenaAgent


class RandomAgent(ArenaAgent):
    def __init__(self, num_actions: int, seed: int | None = None, **_: Any) -> None:
        self._num_actions = num_actions
        self._rng = np.random.default_rng(seed)

    def act(self, raw_observation: Any, *, deterministic: bool = False) -> int | list[int]:
        del deterministic
        obs_array = np.asarray(raw_observation, dtype=object)
        if obs_array.ndim >= 1 and obs_array.shape[0] > 1:
            return self._rng.integers(self._num_actions, size=int(obs_array.shape[0])).astype(int).tolist()
        return int(self._rng.integers(self._num_actions))
