from __future__ import annotations

from typing import Any

from .agents.base import ArenaAgent
from .agents.random_agent import RandomAgent
from .agents.yfu_saltyfish import YFuSaltyFishAgent


BUILTIN_AGENTS = {
    "random": RandomAgent,
    "yfu_saltyfish": YFuSaltyFishAgent,
}


def create_agent(agent_type: str, **kwargs: Any) -> ArenaAgent:
    try:
        agent_cls = BUILTIN_AGENTS[agent_type]
    except KeyError as exc:
        known = ", ".join(sorted(BUILTIN_AGENTS))
        raise ValueError(f"Unknown agent type '{agent_type}'. Known types: {known}.") from exc
    return agent_cls(**kwargs)
