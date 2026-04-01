from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class ArenaAgent(ABC):
    """Common interface for head-to-head arena agents."""

    def reset(self) -> None:
        """Reset any per-episode state."""

    @abstractmethod
    def act(self, raw_observation: Any, *, deterministic: bool = False) -> int | list[int]:
        """Return one or more environment actions in the active football action set."""

    def close(self) -> None:
        """Release resources when the match is over."""
