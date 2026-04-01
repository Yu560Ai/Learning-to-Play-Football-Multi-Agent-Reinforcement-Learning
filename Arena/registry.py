from __future__ import annotations

from typing import Any

from .agents.base import ArenaAgent
from .agents.google_builtin import GoogleBuiltinAgent
from .agents.random_agent import RandomAgent
from .agents.yfu_multiagent import YFuMultiAgent
from .agents.yfu_saltyfish import YFuSaltyFishAgent


BUILTIN_AGENTS = {
    "google_builtin": GoogleBuiltinAgent,
    "random": RandomAgent,
    "yfu_multiagent": YFuMultiAgent,
    "yfu_saltyfish": YFuSaltyFishAgent,
}


def create_agent(agent_type: str, **kwargs: Any) -> ArenaAgent:
    try:
        agent_cls = BUILTIN_AGENTS[agent_type]
    except KeyError as exc:
        known = ", ".join(sorted(BUILTIN_AGENTS))
        raise ValueError(f"Unknown agent type '{agent_type}'. Known types: {known}.") from exc
    return agent_cls(**kwargs)
