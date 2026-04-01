from __future__ import annotations

from typing import Any

import numpy as np

from .base import ArenaAgent


class GoogleBuiltinAgent(ArenaAgent):
    """Delegate control to the environment's built-in football AI."""

    BUILTIN_AI_ACTION = 19

    def __init__(self, num_actions: int, **_: Any) -> None:
        if num_actions <= self.BUILTIN_AI_ACTION:
            raise ValueError(
                "GoogleBuiltinAgent requires an action set that includes action_builtin_ai "
                "(use action_set='v2' or 'full')."
            )
        self._num_actions = num_actions

    def act(self, raw_observation: Any, *, deterministic: bool = False) -> int | list[int]:
        del deterministic
        obs_array = np.asarray(raw_observation, dtype=object)
        if obs_array.ndim == 0:
            return self.BUILTIN_AI_ACTION
        if obs_array.ndim >= 1 and obs_array.shape[0] > 1:
            return [self.BUILTIN_AI_ACTION] * int(obs_array.shape[0])
        return self.BUILTIN_AI_ACTION
