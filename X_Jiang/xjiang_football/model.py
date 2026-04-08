from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.distributions import Categorical

from xjiang_football.features import SIMPLE115_DIM, split_simple115


def _orthogonal_init(module: nn.Module, std: float = 1.0, bias_const: float = 0.0) -> nn.Module:
    if isinstance(module, nn.Linear):
        nn.init.orthogonal_(module.weight, std)
        if module.bias is not None:
            nn.init.constant_(module.bias, bias_const)
    return module


def _mlp(input_dim: int, hidden_dim: int) -> nn.Sequential:
    return nn.Sequential(
        _orthogonal_init(nn.Linear(input_dim, hidden_dim), std=2**0.5),
        nn.Tanh(),
        _orthogonal_init(nn.Linear(hidden_dim, hidden_dim), std=2**0.5),
        nn.Tanh(),
    )


@dataclass(frozen=True)
class ModelConfig:
    head_dim: int = 64
    trunk_dim: int = 256
    critic_dim: int = 256


class StructuredFeatureEncoder(nn.Module):
    def __init__(self, obs_dim: int, config: ModelConfig) -> None:
        super().__init__()
        if obs_dim < SIMPLE115_DIM:
            raise ValueError(f"Expected at least {SIMPLE115_DIM} observation features, got {obs_dim}.")
        self.extra_dim = obs_dim - SIMPLE115_DIM
        head_dim = config.head_dim
        trunk_dim = config.trunk_dim

        self.left_team_head = _mlp(22, head_dim)
        self.left_dir_head = _mlp(22, head_dim)
        self.right_team_head = _mlp(22, head_dim)
        self.right_dir_head = _mlp(22, head_dim)
        self.ball_head = _mlp(9, head_dim)
        self.active_head = _mlp(11, head_dim)
        self.game_mode_head = _mlp(7, head_dim)
        self.extra_head = _mlp(self.extra_dim, head_dim) if self.extra_dim > 0 else None

        merged_dim = head_dim * (7 + int(self.extra_head is not None))
        self.trunk = nn.Sequential(
            _orthogonal_init(nn.Linear(merged_dim, trunk_dim), std=2**0.5),
            nn.LayerNorm(trunk_dim),
            nn.Tanh(),
            _orthogonal_init(nn.Linear(trunk_dim, trunk_dim), std=2**0.5),
            nn.Tanh(),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        base_obs = obs[:, :SIMPLE115_DIM]
        parts = split_simple115(base_obs)
        features = [
            self.left_team_head(parts["left_team"]),
            self.left_dir_head(parts["left_dir"]),
            self.right_team_head(parts["right_team"]),
            self.right_dir_head(parts["right_dir"]),
            self.ball_head(parts["ball_state"]),
            self.active_head(parts["active"]),
            self.game_mode_head(parts["game_mode"]),
        ]
        if self.extra_head is not None:
            features.append(self.extra_head(obs[:, SIMPLE115_DIM:]))
        return self.trunk(torch.cat(features, dim=-1))


class ActorCritic(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        num_players: int,
        config: ModelConfig | None = None,
    ) -> None:
        super().__init__()
        cfg = config or ModelConfig()
        self.actor_encoder = StructuredFeatureEncoder(obs_dim, cfg)
        self.policy_head = _orthogonal_init(nn.Linear(cfg.trunk_dim, action_dim), std=0.01)
        self.value_head = _orthogonal_init(nn.Linear(cfg.trunk_dim, 1), std=1.0)

    def actor_forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        actor_features = self.actor_encoder(obs)
        return self.policy_head(actor_features), actor_features

    def get_value(self, obs: torch.Tensor, global_obs: torch.Tensor | None = None) -> torch.Tensor:
        actor_features = self.actor_encoder(obs)
        return self.value_head(actor_features).squeeze(-1)

    def get_action_and_value(
        self,
        obs: torch.Tensor,
        global_obs: torch.Tensor | None = None,
        action: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        logits, actor_features = self.actor_forward(obs)
        dist = Categorical(logits=logits)
        if action is None:
            action = dist.sample()
        value = self.value_head(actor_features).squeeze(-1)
        return action, dist.log_prob(action), dist.entropy(), value

    def act(
        self,
        obs: torch.Tensor,
        global_obs: torch.Tensor | None = None,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits, actor_features = self.actor_forward(obs)
        dist = Categorical(logits=logits)
        action = torch.argmax(logits, dim=-1) if deterministic else dist.sample()
        value = self.value_head(actor_features).squeeze(-1)
        return action, dist.log_prob(action), value
