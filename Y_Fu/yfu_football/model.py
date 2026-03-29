from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from torch import nn
from torch.distributions import Categorical


MODEL_TYPES = ("auto", "mlp", "residual_mlp", "separate_mlp", "cnn")


def _orthogonal_init(module: nn.Module, std: float = 1.0, bias_const: float = 0.0) -> nn.Module:
    if isinstance(module, (nn.Linear, nn.Conv2d)):
        nn.init.orthogonal_(module.weight, std)
        if module.bias is not None:
            nn.init.constant_(module.bias, bias_const)
    return module


def _activation() -> nn.Module:
    return nn.Tanh()


def _build_mlp(input_dim: int, hidden_sizes: Sequence[int], *, layer_norm: bool = False) -> tuple[nn.Sequential, int]:
    layers: list[nn.Module] = []
    last_dim = input_dim
    for hidden_size in hidden_sizes:
        layers.append(_orthogonal_init(nn.Linear(last_dim, hidden_size), std=2**0.5))
        if layer_norm:
            layers.append(nn.LayerNorm(hidden_size))
        layers.append(_activation())
        last_dim = hidden_size
    return nn.Sequential(*layers), last_dim


class ResidualBlock(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.fc1 = _orthogonal_init(nn.Linear(width, width), std=2**0.5)
        self.fc2 = _orthogonal_init(nn.Linear(width, width), std=2**0.5)
        self.norm1 = nn.LayerNorm(width)
        self.norm2 = nn.LayerNorm(width)
        self.act = _activation()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.fc1(x)
        x = self.norm1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.norm2(x)
        x = x + residual
        return self.act(x)


class ResidualMLPEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_sizes: Sequence[int]) -> None:
        super().__init__()
        if not hidden_sizes:
            hidden_sizes = (256, 256)
        width = hidden_sizes[0]
        self.input_layer = _orthogonal_init(nn.Linear(input_dim, width), std=2**0.5)
        self.input_norm = nn.LayerNorm(width)
        self.act = _activation()

        blocks: list[nn.Module] = []
        last_dim = width
        for hidden_size in hidden_sizes[1:]:
            if hidden_size == last_dim:
                blocks.append(ResidualBlock(last_dim))
            else:
                blocks.append(_orthogonal_init(nn.Linear(last_dim, hidden_size), std=2**0.5))
                blocks.append(nn.LayerNorm(hidden_size))
                blocks.append(_activation())
                last_dim = hidden_size
        self.body = nn.Sequential(*blocks)
        self.output_dim = last_dim

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        x = self.input_layer(obs)
        x = self.input_norm(x)
        x = self.act(x)
        return self.body(x)


class ConvEncoder(nn.Module):
    def __init__(self, obs_shape: Sequence[int], feature_dim: int) -> None:
        super().__init__()
        if len(obs_shape) == 2:
            height, width = int(obs_shape[0]), int(obs_shape[1])
            channels = 1
        elif len(obs_shape) == 3:
            height, width, channels = (int(obs_shape[0]), int(obs_shape[1]), int(obs_shape[2]))
        else:
            raise ValueError(f"CNN encoder expects 2D or 3D observations, got {tuple(obs_shape)!r}")

        self.obs_shape = tuple(int(dim) for dim in obs_shape)
        self.channels = channels
        self.conv = nn.Sequential(
            _orthogonal_init(nn.Conv2d(channels, 32, kernel_size=5, stride=2, padding=2), std=2**0.5),
            nn.ReLU(),
            _orthogonal_init(nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), std=2**0.5),
            nn.ReLU(),
            _orthogonal_init(nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1), std=2**0.5),
            nn.ReLU(),
        )

        with torch.no_grad():
            dummy = torch.zeros(1, channels, height, width)
            conv_out = self.conv(dummy)
            flat_dim = int(math.prod(conv_out.shape[1:]))

        self.proj = nn.Sequential(
            nn.Flatten(),
            _orthogonal_init(nn.Linear(flat_dim, feature_dim), std=2**0.5),
            nn.LayerNorm(feature_dim),
            nn.ReLU(),
        )
        self.output_dim = feature_dim

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        if len(self.obs_shape) == 2:
            obs = obs.view(-1, 1, self.obs_shape[0], self.obs_shape[1])
        else:
            obs = obs.view(-1, self.obs_shape[0], self.obs_shape[1], self.obs_shape[2]).permute(0, 3, 1, 2)
        return self.proj(self.conv(obs))


def _resolve_model_type(model_type: str, obs_shape: Sequence[int]) -> str:
    if model_type == "auto":
        return "cnn" if len(obs_shape) >= 2 else "residual_mlp"
    if model_type not in MODEL_TYPES:
        raise ValueError(f"Unsupported model_type={model_type!r}. Expected one of {MODEL_TYPES}.")
    return model_type


class ActorCritic(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_sizes: Sequence[int] = (256, 256),
        obs_shape: Sequence[int] | None = None,
        model_type: str = "auto",
        feature_dim: int = 256,
    ) -> None:
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.obs_shape = tuple(int(dim) for dim in (obs_shape or (obs_dim,)))
        self.model_type = _resolve_model_type(model_type, self.obs_shape)

        if self.model_type == "cnn":
            self.encoder = ConvEncoder(self.obs_shape, feature_dim=feature_dim)
            actor_input_dim = self.encoder.output_dim
            critic_input_dim = self.encoder.output_dim
            self.actor_body, actor_body_dim = _build_mlp(actor_input_dim, hidden_sizes, layer_norm=True)
            self.critic_body, critic_body_dim = _build_mlp(critic_input_dim, hidden_sizes, layer_norm=True)
            self._shared_encoder = True
        elif self.model_type == "separate_mlp":
            self.actor_encoder, actor_input_dim = _build_mlp(self.obs_dim, hidden_sizes, layer_norm=True)
            self.critic_encoder, critic_input_dim = _build_mlp(self.obs_dim, hidden_sizes, layer_norm=True)
            self.actor_body = nn.Identity()
            self.critic_body = nn.Identity()
            actor_body_dim = actor_input_dim
            critic_body_dim = critic_input_dim
            self._shared_encoder = False
        elif self.model_type == "residual_mlp":
            self.encoder = ResidualMLPEncoder(self.obs_dim, hidden_sizes)
            actor_input_dim = self.encoder.output_dim
            critic_input_dim = self.encoder.output_dim
            self.actor_body = nn.Identity()
            self.critic_body = nn.Identity()
            actor_body_dim = actor_input_dim
            critic_body_dim = critic_input_dim
            self._shared_encoder = True
        else:
            self.encoder, encoder_out_dim = _build_mlp(self.obs_dim, hidden_sizes, layer_norm=False)
            actor_input_dim = encoder_out_dim
            critic_input_dim = encoder_out_dim
            self.actor_body = nn.Identity()
            self.critic_body = nn.Identity()
            actor_body_dim = actor_input_dim
            critic_body_dim = critic_input_dim
            self._shared_encoder = True

        self.policy_head = _orthogonal_init(nn.Linear(actor_body_dim, action_dim), std=0.01)
        self.value_head = _orthogonal_init(nn.Linear(critic_body_dim, 1), std=1.0)

    def _encode(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.model_type == "separate_mlp":
            return self.actor_encoder(obs), self.critic_encoder(obs)
        shared = self.encoder(obs)
        return shared, shared

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        actor_features, critic_features = self._encode(obs)
        actor_features = self.actor_body(actor_features)
        critic_features = self.critic_body(critic_features)
        return self.policy_head(actor_features), self.value_head(critic_features).squeeze(-1)

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
