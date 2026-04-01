from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

from .model import ConvEncoder, ResidualMLPEncoder, _build_mlp, _orthogonal_init


@dataclass
class IQLConfig:
    obs_dim: int = 1764
    action_dim: int = 19
    obs_shape: tuple[int, ...] = (42, 42)
    hidden_sizes: tuple[int, ...] = (256, 256)
    feature_dim: int = 256
    encoder_type: str = "cnn"
    env_name: str = "5_vs_5"
    representation: str = "extracted"
    rewards: str = "scoring,checkpoints"
    num_controlled_players: int = 4
    channel_dimensions: tuple[int, int] = (42, 42)
    pass_success_reward: float = 0.0
    pass_failure_penalty: float = 0.0
    pass_progress_reward_scale: float = 0.0
    shot_attempt_reward: float = 0.0
    attacking_possession_reward: float = 0.0
    attacking_x_threshold: float = 0.55
    final_third_entry_reward: float = 0.0
    possession_retention_reward: float = 0.0
    possession_recovery_reward: float = 0.0
    defensive_third_recovery_reward: float = 0.0
    opponent_attacking_possession_penalty: float = 0.0
    own_half_turnover_penalty: float = 0.0
    own_half_x_threshold: float = 0.0
    defensive_x_threshold: float = -0.45
    pending_pass_horizon: int = 8
    learning_rate: float = 3e-4
    gamma: float = 0.993
    expectile: float = 0.7
    temperature: float = 3.0
    tau: float = 0.005
    batch_size: int = 4096
    total_gradient_steps: int = 1_000_000
    eval_interval: int = 10_000
    reward_normalization_enabled: bool = False
    reward_normalization_scale: float = 1.0
    reward_normalization_source: str = "none"
    seed: int = 42


def _build_encoder(
    *,
    obs_dim: int,
    obs_shape: tuple[int, ...],
    feature_dim: int,
    hidden_sizes: tuple[int, ...],
    encoder_type: str,
) -> tuple[nn.Module, int]:
    if encoder_type == "cnn":
        encoder = ConvEncoder(obs_shape=obs_shape, feature_dim=feature_dim)
        return encoder, encoder.output_dim
    if encoder_type == "residual_mlp":
        encoder = ResidualMLPEncoder(input_dim=obs_dim, hidden_sizes=hidden_sizes)
        return encoder, encoder.output_dim
    encoder, output_dim = _build_mlp(obs_dim, hidden_sizes, layer_norm=True)
    return encoder, output_dim


class QNetwork(nn.Module):
    def __init__(
        self,
        *,
        obs_dim: int,
        action_dim: int,
        obs_shape: tuple[int, ...],
        hidden_sizes: tuple[int, ...],
        feature_dim: int,
        encoder_type: str,
    ) -> None:
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.obs_shape = tuple(int(dim) for dim in obs_shape)
        self.encoder_type = encoder_type
        self.encoder, encoder_out_dim = _build_encoder(
            obs_dim=self.obs_dim,
            obs_shape=self.obs_shape,
            feature_dim=feature_dim,
            hidden_sizes=hidden_sizes,
            encoder_type=encoder_type,
        )
        self.body, body_dim = _build_mlp(encoder_out_dim, hidden_sizes, layer_norm=True)
        self.output_head = _orthogonal_init(nn.Linear(body_dim, self.action_dim), std=1.0)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(obs)
        features = self.body(encoded)
        return self.output_head(features)


class VNetwork(nn.Module):
    def __init__(
        self,
        *,
        obs_dim: int,
        obs_shape: tuple[int, ...],
        hidden_sizes: tuple[int, ...],
        feature_dim: int,
        encoder_type: str,
    ) -> None:
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.obs_shape = tuple(int(dim) for dim in obs_shape)
        self.encoder_type = encoder_type
        self.encoder, encoder_out_dim = _build_encoder(
            obs_dim=self.obs_dim,
            obs_shape=self.obs_shape,
            feature_dim=feature_dim,
            hidden_sizes=hidden_sizes,
            encoder_type=encoder_type,
        )
        self.body, body_dim = _build_mlp(encoder_out_dim, hidden_sizes, layer_norm=True)
        self.output_head = _orthogonal_init(nn.Linear(body_dim, 1), std=1.0)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(obs)
        features = self.body(encoded)
        return self.output_head(features).squeeze(-1)


class DiscreteIQL:
    def __init__(self, config: IQLConfig, device: torch.device | str = "cpu") -> None:
        self.config = config
        self.device = torch.device(device)

        network_kwargs = {
            "obs_dim": config.obs_dim,
            "action_dim": config.action_dim,
            "obs_shape": config.obs_shape,
            "hidden_sizes": config.hidden_sizes,
            "feature_dim": config.feature_dim,
            "encoder_type": config.encoder_type,
        }
        value_kwargs = {
            "obs_dim": config.obs_dim,
            "obs_shape": config.obs_shape,
            "hidden_sizes": config.hidden_sizes,
            "feature_dim": config.feature_dim,
            "encoder_type": config.encoder_type,
        }

        self.q1 = QNetwork(**network_kwargs).to(self.device)
        self.q2 = QNetwork(**network_kwargs).to(self.device)
        self.q1_target = QNetwork(**network_kwargs).to(self.device)
        self.q2_target = QNetwork(**network_kwargs).to(self.device)
        self.v = VNetwork(**value_kwargs).to(self.device)

        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())
        self.q1_target.eval()
        self.q2_target.eval()
        for parameter in self.q1_target.parameters():
            parameter.requires_grad_(False)
        for parameter in self.q2_target.parameters():
            parameter.requires_grad_(False)

        self.q1_optimizer = torch.optim.Adam(self.q1.parameters(), lr=config.learning_rate, eps=1e-5)
        self.q2_optimizer = torch.optim.Adam(self.q2.parameters(), lr=config.learning_rate, eps=1e-5)
        self.v_optimizer = torch.optim.Adam(self.v.parameters(), lr=config.learning_rate, eps=1e-5)

    def _soft_update_targets(self) -> None:
        tau = float(self.config.tau)
        with torch.no_grad():
            for target_parameter, parameter in zip(self.q1_target.parameters(), self.q1.parameters(), strict=True):
                target_parameter.data.mul_(1.0 - tau).add_(parameter.data, alpha=tau)
            for target_parameter, parameter in zip(self.q2_target.parameters(), self.q2.parameters(), strict=True):
                target_parameter.data.mul_(1.0 - tau).add_(parameter.data, alpha=tau)

    def update(self, batch: dict[str, torch.Tensor]) -> dict[str, float]:
        obs = batch["obs"]
        next_obs = batch["next_obs"]
        actions = batch["action"].long()
        rewards = batch["reward"]
        done = batch["done"].float()
        timeout = batch["timeout"].float()
        terminal = done * (1.0 - timeout)

        with torch.no_grad():
            target_q1 = self.q1_target(obs)
            target_q2 = self.q2_target(obs)
            target_q = torch.minimum(target_q1, target_q2).gather(1, actions.unsqueeze(-1)).squeeze(-1)

        v_values = self.v(obs)
        advantages = target_q - v_values
        expectile_weight = torch.where(
            advantages > 0.0,
            torch.full_like(advantages, float(self.config.expectile)),
            torch.full_like(advantages, 1.0 - float(self.config.expectile)),
        )
        v_loss = (expectile_weight * advantages.pow(2)).mean()

        self.v_optimizer.zero_grad()
        v_loss.backward()
        self.v_optimizer.step()

        with torch.no_grad():
            next_v = self.v(next_obs)
            q_target = rewards + float(self.config.gamma) * (1.0 - terminal) * next_v

        q1_pred = self.q1(obs).gather(1, actions.unsqueeze(-1)).squeeze(-1)
        q2_pred = self.q2(obs).gather(1, actions.unsqueeze(-1)).squeeze(-1)
        q1_loss = (q1_pred - q_target).pow(2).mean()
        q2_loss = (q2_pred - q_target).pow(2).mean()

        self.q1_optimizer.zero_grad()
        q1_loss.backward()
        self.q1_optimizer.step()

        self.q2_optimizer.zero_grad()
        q2_loss.backward()
        self.q2_optimizer.step()

        self._soft_update_targets()

        return {
            "v_loss": float(v_loss.item()),
            "q1_loss": float(q1_loss.item()),
            "q2_loss": float(q2_loss.item()),
            "q_loss": float(0.5 * (q1_loss.item() + q2_loss.item())),
            "mean_q": float(0.5 * (q1_pred.mean().item() + q2_pred.mean().item())),
            "mean_v": float(v_values.mean().item()),
            "mean_target_q": float(q_target.mean().item()),
            "mean_advantage": float(advantages.mean().item()),
        }

    def act(self, obs: torch.Tensor | np.ndarray, deterministic: bool = False) -> torch.Tensor:
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        if obs_tensor.ndim == 1:
            obs_tensor = obs_tensor.unsqueeze(0)
        with torch.no_grad():
            q_values = self.policy_logits(obs_tensor)
            if deterministic:
                return torch.argmax(q_values, dim=-1)
            distribution = Categorical(logits=q_values / max(float(self.config.temperature), 1e-6))
            return distribution.sample()

    def policy_logits(self, obs: torch.Tensor | np.ndarray) -> torch.Tensor:
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        if obs_tensor.ndim == 1:
            obs_tensor = obs_tensor.unsqueeze(0)
        with torch.no_grad():
            return torch.minimum(self.q1(obs_tensor), self.q2(obs_tensor))

    def save_checkpoint(self, path: str | Path) -> Path:
        checkpoint_path = Path(path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "config": asdict(self.config),
                "q1_state_dict": self.q1.state_dict(),
                "q2_state_dict": self.q2.state_dict(),
                "q1_target_state_dict": self.q1_target.state_dict(),
                "q2_target_state_dict": self.q2_target.state_dict(),
                "v_state_dict": self.v.state_dict(),
                "q1_optimizer_state_dict": self.q1_optimizer.state_dict(),
                "q2_optimizer_state_dict": self.q2_optimizer.state_dict(),
                "v_optimizer_state_dict": self.v_optimizer.state_dict(),
            },
            checkpoint_path,
        )
        return checkpoint_path

    @classmethod
    def load_checkpoint(
        cls,
        path: str | Path,
        device: torch.device | str = "cpu",
    ) -> "DiscreteIQL":
        checkpoint = torch.load(path, map_location="cpu")
        config = IQLConfig(**checkpoint["config"])
        iql = cls(config=config, device=device)
        iql.q1.load_state_dict(checkpoint["q1_state_dict"])
        iql.q2.load_state_dict(checkpoint["q2_state_dict"])
        iql.q1_target.load_state_dict(checkpoint["q1_target_state_dict"])
        iql.q2_target.load_state_dict(checkpoint["q2_target_state_dict"])
        iql.v.load_state_dict(checkpoint["v_state_dict"])
        if "q1_optimizer_state_dict" in checkpoint:
            iql.q1_optimizer.load_state_dict(checkpoint["q1_optimizer_state_dict"])
        if "q2_optimizer_state_dict" in checkpoint:
            iql.q2_optimizer.load_state_dict(checkpoint["q2_optimizer_state_dict"])
        if "v_optimizer_state_dict" in checkpoint:
            iql.v_optimizer.load_state_dict(checkpoint["v_optimizer_state_dict"])
        return iql
