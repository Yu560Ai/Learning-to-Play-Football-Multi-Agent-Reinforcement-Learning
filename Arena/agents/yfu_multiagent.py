from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_Y_FU_ROOT = _PROJECT_ROOT / "Y_Fu"
if str(_Y_FU_ROOT) not in sys.path:
    sys.path.insert(0, str(_Y_FU_ROOT))

from Y_Fu.yfu_football.model import ActorCritic

from .base import ArenaAgent


def _resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


class YFuMultiAgent(ArenaAgent):
    def __init__(self, checkpoint: str, device: str = "cpu", **_: Any) -> None:
        if not checkpoint:
            raise ValueError("YFuMultiAgent requires a checkpoint path.")
        self.device = _resolve_device(device)
        self.checkpoint = torch.load(checkpoint, map_location="cpu", weights_only=False)
        config = self.checkpoint.get("config", {})
        self.obs_dim = int(self.checkpoint["obs_dim"])
        self.num_players = int(self.checkpoint.get("num_players", config.get("num_controlled_players", 1)))
        self.model = ActorCritic(
            obs_dim=self.obs_dim,
            action_dim=int(self.checkpoint["action_dim"]),
            hidden_sizes=tuple(config.get("hidden_sizes", [256, 256])),
            obs_shape=tuple(self.checkpoint.get("obs_shape", (self.obs_dim,))),
            model_type=config.get("model_type", "auto"),
            feature_dim=int(config.get("feature_dim", 256)),
        )
        self.model.load_state_dict(self.checkpoint["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()

    def act(self, raw_observation: Any, *, deterministic: bool = False) -> int | list[int]:
        obs = np.asarray(raw_observation, dtype=np.float32)
        if obs.ndim == 0:
            raise ValueError("YFuMultiAgent received an empty observation.")
        if obs.ndim == 1:
            obs = obs.reshape(1, self.obs_dim)
        else:
            obs = obs.reshape(obs.shape[0], self.obs_dim)
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            action, _, _ = self.model.act(obs_tensor, deterministic=deterministic)
        action_np = action.cpu().numpy().astype(np.int64)
        if action_np.size == 1:
            return int(action_np.item())
        return action_np.tolist()
