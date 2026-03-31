from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class TrainConfig:
    # environment
    env_name: str = "5_vs_5"
    representation: str = "extracted"
    rewards: str = "scoring,checkpoints"
    num_controlled_players: int = 4
    channel_dimensions: tuple[int, int] = (42, 42)
    render: bool = False

    # model
    model_type: str = "cnn"
    feature_dim: int = 256
    hidden_sizes: tuple[int, ...] = (256, 256)

    # ppo
    total_timesteps: int = 20_000
    rollout_steps: int = 128
    update_epochs: int = 4
    num_minibatches: int = 4
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5

    # misc
    seed: int = 42
    save_interval: int = 2
    log_interval: int = 1
    save_dir: str = "X_Jiang/checkpoints"
    logdir: str = "X_Jiang/logs"
    device: str = "auto"
    init_checkpoint: str | None = None

    # reward shaping
    use_reward_shaping: bool = False
    pass_success_reward: float = 0.0
    pass_failure_penalty: float = 0.0
    pass_progress_reward_scale: float = 0.0
    shot_attempt_reward: float = 0.0
    attacking_possession_reward: float = 0.0
    attacking_x_threshold: float = 0.55
    final_third_entry_reward: float = 0.0
    possession_retention_reward: float = 0.0
    own_half_turnover_penalty: float = 0.0
    own_half_x_threshold: float = 0.0
    pending_pass_horizon: int = 8
    defensive_ball_chasing_reward_scale: float = 0.0
    goalkeeper_overextension_penalty: float = 0.0
    goalkeeper_max_x: float = -0.35
    defensive_recovery_reward: float = 0.0
    defensive_x_threshold: float = 0.15


PRESETS: dict[str, dict[str, Any]] = {
    "five_v_five_debug": {
        "env_name": "5_vs_5",
        "representation": "extracted",
        "rewards": "scoring,checkpoints",
        "num_controlled_players": 4,
        "channel_dimensions": (42, 42),
        "model_type": "cnn",
        "feature_dim": 256,
        "hidden_sizes": (128, 128),
        "total_timesteps": 20_000,
        "rollout_steps": 128,
        "update_epochs": 3,
        "num_minibatches": 4,
        "learning_rate": 3e-4,
        "save_interval": 2,
        "log_interval": 1,
        "save_dir": "X_Jiang/checkpoints/five_v_five_debug",
        "logdir": "X_Jiang/logs/five_v_five_debug",
        "device": "auto",
        "use_reward_shaping": False,
    },
    "five_v_five_base": {
        "env_name": "5_vs_5",
        "representation": "extracted",
        "rewards": "scoring,checkpoints",
        "num_controlled_players": 4,
        "channel_dimensions": (42, 42),
        "model_type": "cnn",
        "feature_dim": 256,
        "hidden_sizes": (256, 256),
        "total_timesteps": 400_000,
        "rollout_steps": 512,
        "update_epochs": 4,
        "num_minibatches": 4,
        "learning_rate": 2.5e-4,
        "save_interval": 10,
        "log_interval": 1,
        "save_dir": "X_Jiang/checkpoints/five_v_five_base",
        "logdir": "X_Jiang/logs/five_v_five_base",
        "device": "auto",
        "use_reward_shaping": True,
        "defensive_ball_chasing_reward_scale": 0.03,
        "goalkeeper_overextension_penalty": 0.08,
        "goalkeeper_max_x": -0.35,
        "defensive_recovery_reward": 0.04,
        "defensive_x_threshold": 0.15,
        "pass_success_reward": 0.06,
        "pass_failure_penalty": 0.03,
        "pass_progress_reward_scale": 0.04,
        "shot_attempt_reward": 0.02,
        "attacking_possession_reward": 0.0005,
        "attacking_x_threshold": 0.55,
        "final_third_entry_reward": 0.02,
        "possession_retention_reward": 0.0005,
        "own_half_turnover_penalty": 0.02,
        "own_half_x_threshold": 0.0,
        "pending_pass_horizon": 8,
    },
}


def build_config(preset_name: str, overrides: dict[str, Any] | None = None) -> TrainConfig:
    if preset_name not in PRESETS:
        raise ValueError(f"Unknown preset: {preset_name}. Available presets: {list(PRESETS)}")

    merged: dict[str, Any] = dict(PRESETS[preset_name])
    if overrides:
        merged.update(overrides)

    return TrainConfig(**merged)