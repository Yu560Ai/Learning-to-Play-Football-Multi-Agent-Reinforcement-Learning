from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn
from torch.distributions import Categorical


def _orthogonal_init(module: nn.Module, std: float = 1.0, bias_const: float = 0.0) -> nn.Module:
    if isinstance(module, nn.Linear):
        nn.init.orthogonal_(module.weight, std)
        nn.init.constant_(module.bias, bias_const)
    return module


class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden_sizes: Sequence[int] = (256, 256)) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        last_dim = obs_dim
        for hidden_size in hidden_sizes:
            layers.append(_orthogonal_init(nn.Linear(last_dim, hidden_size), std=2**0.5))
            layers.append(nn.Tanh())
            last_dim = hidden_size
        self.backbone = nn.Sequential(*layers)
        self.policy_head = _orthogonal_init(nn.Linear(last_dim, action_dim), std=0.01)
        self.value_head = _orthogonal_init(nn.Linear(last_dim, 1), std=1.0)

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.backbone(obs)
        return self.policy_head(features), self.value_head(features).squeeze(-1)

    def get_value(self, obs: torch.Tensor) -> torch.Tensor:
        _, value = self.forward(obs)
        return value

    def get_action_and_value(
        self,
        obs: torch.Tensor,
        action: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        logits, value = self.forward(obs)
        distribution = Categorical(logits=logits)
        if action is None:
            action = distribution.sample()
        log_prob = distribution.log_prob(action)
        entropy = distribution.entropy()
        return action, log_prob, entropy, value

    def act(
        self,
        obs: torch.Tensor,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits, value = self.forward(obs)
        distribution = Categorical(logits=logits)
        action = torch.argmax(logits, dim=-1) if deterministic else distribution.sample()
        log_prob = distribution.log_prob(action)
        return action, log_prob, value
