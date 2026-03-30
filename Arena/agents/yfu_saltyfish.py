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

from Y_Fu.saltyfish_baseline.env import REDUCED_ACTION_INDICES
from Y_Fu.saltyfish_baseline.features import build_engineered_features
from Y_Fu.saltyfish_baseline.model import SaltyFishModelConfig, StructuredSimple115ActorCritic
from Y_Fu.yfu_football.envs import ensure_local_gfootball_path

from .base import ArenaAgent

ensure_local_gfootball_path()
from gfootball.env import wrappers  # type: ignore  # noqa: E402


def _resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


class YFuSaltyFishAgent(ArenaAgent):
    def __init__(self, checkpoint: str, device: str = "cpu", **_: Any) -> None:
        if not checkpoint:
            raise ValueError("YFuSaltyFishAgent requires a checkpoint path.")
        self.device = _resolve_device(device)
        self.checkpoint = torch.load(checkpoint, map_location="cpu", weights_only=False)
        config = self.checkpoint.get("config", {})
        self.use_engineered_features = bool(config.get("use_engineered_features", False))
        self.action_map = np.asarray(
            self.checkpoint.get("action_map", list(REDUCED_ACTION_INDICES)),
            dtype=np.int64,
        )
        self.model = StructuredSimple115ActorCritic(
            obs_dim=int(self.checkpoint["obs_dim"]),
            action_dim=int(self.checkpoint["action_dim"]),
            config=SaltyFishModelConfig(
                head_dim=int(config.get("head_dim", 64)),
                trunk_dim=int(config.get("trunk_dim", 256)),
            ),
        )
        self.model.load_state_dict(self.checkpoint["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()

    def act(self, raw_observation: dict[str, Any], *, deterministic: bool = False) -> int:
        simple115 = wrappers.Simple115StateWrapper.convert_observation([raw_observation], True).astype(np.float32)
        if self.use_engineered_features:
            engineered = build_engineered_features(raw_observation).reshape(1, -1)
            obs = np.concatenate([simple115, engineered], axis=1)
        else:
            obs = simple115
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            action, _, _ = self.model.act(obs_tensor, deterministic=deterministic)
        action_index = int(action.item())
        return int(self.action_map[action_index])
