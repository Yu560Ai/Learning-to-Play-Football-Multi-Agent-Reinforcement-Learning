from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class TrainConfig:
    env_name: str = "5_vs_5"
    representation: str = "extracted"
    rewards: str = "scoring,checkpoints"
    num_controlled_players: int = 4
    channel_dimensions: tuple[int, int] = (42, 42)
    render: bool = False

    model_type: str = "tactical_mlp"
    feature_dim: int = 256
    hidden_sizes: tuple[int, ...] = (256, 256)
    use_specialized_policy_heads: bool = True
    use_specialized_value_heads: bool = True

    total_timesteps: int = 400000
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

    seed: int = 42
    save_interval: int = 5
    log_interval: int = 1
    save_dir: str = "X_Jiang/checkpoints"
    logdir: str = "X_Jiang/logs"
    device: str = "auto"
    init_checkpoint: str | None = None

    use_reward_shaping: bool = True
    closest_player_to_ball_reward: float = 0.08
    first_defender_pressure_reward: float = 0.06
    second_player_support_reward: float = 0.05
    recover_shape_reward: float = 0.04
    hold_shape_reward: float = 0.02
    ball_watch_penalty: float = 0.06
    idle_wander_penalty: float = 0.03
    goalkeeper_home_reward: float = 0.03
    goalkeeper_wander_penalty: float = 0.08
    possession_support_reward: float = 0.03
    attack_space_reward: float = 0.04
    progressive_pass_choice_reward: float = 0.05
    progressive_pass_result_reward_scale: float = 0.10
    carry_progress_reward_scale: float = 0.06
    zone_entry_progress_reward: float = 0.10
    safe_reset_pass_reward: float = 0.01
    backward_gk_pass_penalty: float = 0.08
    unnecessary_goalkeeper_reset_penalty: float = 0.12
    non_emergency_clear_penalty: float = 0.10
    shot_choice_reward: float = 0.05
    missed_shot_window_penalty: float = 0.10
    shot_execution_reward: float = 0.15
    on_ball_stall_penalty: float = 0.08
    on_ball_backward_drift_penalty: float = 0.07
    on_ball_lateral_zigzag_penalty: float = 0.05
    support_spacing_reward: float = 0.05
    support_spacing_penalty: float = 0.04
    support_forward_lane_reward: float = 0.04
    support_static_penalty: float = 0.03
    safe_reset_overuse_penalty: float = 0.10
    support_behind_ball_penalty: float = 0.05
    use_adaptive_reward_weights: bool = True
    adaptive_reward_interval: int = 10
    adaptive_scale_step: float = 0.10
    adaptive_scale_min: float = 0.60
    adaptive_scale_max: float = 2.00
    adaptive_progression_target: float = 0.28
    adaptive_on_ball_fraction_target: float = 0.12
    adaptive_support_cover_fraction_target: float = 0.50
    adaptive_hold_role_fraction_target: float = 0.28


COMMON_TACTICAL: dict[str, Any] = {
    "env_name": "5_vs_5",
    "representation": "extracted",
    "rewards": "scoring,checkpoints",
    "num_controlled_players": 4,
    "channel_dimensions": (42, 42),
    "model_type": "tactical_mlp",
    "feature_dim": 256,
    "hidden_sizes": (256, 256),
    "use_specialized_policy_heads": True,
    "use_specialized_value_heads": True,
    "learning_rate": 2.5e-4,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_coef": 0.2,
    "ent_coef": 0.01,
    "vf_coef": 0.5,
    "max_grad_norm": 0.5,
    "use_reward_shaping": True,
    "closest_player_to_ball_reward": 0.10,
    "first_defender_pressure_reward": 0.07,
    "second_player_support_reward": 0.06,
    "recover_shape_reward": 0.05,
    "hold_shape_reward": 0.025,
    "ball_watch_penalty": 0.08,
    "idle_wander_penalty": 0.04,
    "goalkeeper_home_reward": 0.04,
    "goalkeeper_wander_penalty": 0.10,
    "possession_support_reward": 0.04,
    "attack_space_reward": 0.05,
    "progressive_pass_choice_reward": 0.05,
    "progressive_pass_result_reward_scale": 0.12,
    "carry_progress_reward_scale": 0.08,
    "zone_entry_progress_reward": 0.14,
    "safe_reset_pass_reward": 0.01,
    "backward_gk_pass_penalty": 0.08,
    "unnecessary_goalkeeper_reset_penalty": 0.14,
    "non_emergency_clear_penalty": 0.14,
    "shot_choice_reward": 0.05,
    "missed_shot_window_penalty": 0.12,
    "shot_execution_reward": 0.18,
    "on_ball_stall_penalty": 0.10,
    "on_ball_backward_drift_penalty": 0.08,
    "on_ball_lateral_zigzag_penalty": 0.06,
    "support_spacing_reward": 0.06,
    "support_spacing_penalty": 0.04,
    "support_forward_lane_reward": 0.05,
    "support_static_penalty": 0.03,
    "safe_reset_overuse_penalty": 0.12,
    "support_behind_ball_penalty": 0.06,
    "use_adaptive_reward_weights": True,
    "adaptive_reward_interval": 10,
    "adaptive_scale_step": 0.10,
    "adaptive_scale_min": 0.60,
    "adaptive_scale_max": 2.00,
    "adaptive_progression_target": 0.28,
    "adaptive_on_ball_fraction_target": 0.12,
    "adaptive_support_cover_fraction_target": 0.50,
    "adaptive_hold_role_fraction_target": 0.28,
    "device": "auto",
}

COMMON_ACADEMY_STABLE: dict[str, Any] = {
    **COMMON_TACTICAL,
    "rewards": "scoring",
    "learning_rate": 1e-4,
    "gamma": 0.993,
    "gae_lambda": 0.95,
    "clip_coef": 0.2,
    "ent_coef": 0.005,
    "vf_coef": 0.5,
    "max_grad_norm": 0.5,
    "hidden_sizes": (192, 192),
    "rollout_steps": 512,
    "update_epochs": 4,
    "num_minibatches": 8,
    "save_interval": 20,
    "log_interval": 1,
    "closest_player_to_ball_reward": 0.03,
    "first_defender_pressure_reward": 0.02,
    "second_player_support_reward": 0.015,
    "recover_shape_reward": 0.01,
    "hold_shape_reward": 0.005,
    "ball_watch_penalty": 0.02,
    "idle_wander_penalty": 0.01,
    "goalkeeper_home_reward": 0.01,
    "goalkeeper_wander_penalty": 0.02,
    "possession_support_reward": 0.01,
    "attack_space_reward": 0.015,
    "progressive_pass_choice_reward": 0.02,
    "progressive_pass_result_reward_scale": 0.04,
    "carry_progress_reward_scale": 0.03,
    "zone_entry_progress_reward": 0.05,
    "safe_reset_pass_reward": 0.0,
    "backward_gk_pass_penalty": 0.02,
    "unnecessary_goalkeeper_reset_penalty": 0.04,
    "non_emergency_clear_penalty": 0.03,
    "shot_choice_reward": 0.02,
    "missed_shot_window_penalty": 0.03,
    "shot_execution_reward": 0.06,
    "on_ball_stall_penalty": 0.03,
    "on_ball_backward_drift_penalty": 0.03,
    "on_ball_lateral_zigzag_penalty": 0.02,
    "support_spacing_reward": 0.02,
    "support_spacing_penalty": 0.015,
    "support_forward_lane_reward": 0.02,
    "support_static_penalty": 0.01,
    "safe_reset_overuse_penalty": 0.04,
    "support_behind_ball_penalty": 0.02,
    "use_adaptive_reward_weights": False,
}


PRESETS: dict[str, dict[str, Any]] = {
    "academy_run_to_score_attack_foundation": {
        **COMMON_ACADEMY_STABLE,
        "env_name": "academy_run_to_score_with_keeper",
        "num_controlled_players": 1,
        "total_timesteps": 20_480,
        "save_interval": 10,
        "save_dir": "X_Jiang/checkpoints/academy_run_to_score_attack_foundation",
        "logdir": "X_Jiang/logs/academy_run_to_score_attack_foundation",
        "first_defender_pressure_reward": 0.0,
        "second_player_support_reward": 0.0,
        "recover_shape_reward": 0.0,
        "hold_shape_reward": 0.002,
        "attack_space_reward": 0.0,
        "progressive_pass_choice_reward": 0.0,
        "progressive_pass_result_reward_scale": 0.0,
        "carry_progress_reward_scale": 0.08,
        "zone_entry_progress_reward": 0.12,
        "shot_choice_reward": 0.02,
        "shot_execution_reward": 0.08,
        "safe_reset_pass_reward": 0.0,
        "safe_reset_overuse_penalty": 0.0,
        "support_forward_lane_reward": 0.0,
        "support_behind_ball_penalty": 0.0,
        "support_spacing_reward": 0.0,
        "support_spacing_penalty": 0.0,
        "support_static_penalty": 0.0,
        "possession_support_reward": 0.0,
        "adaptive_progression_target": 0.45,
    },
    "academy_run_pass_and_shoot_attack_support": {
        **COMMON_ACADEMY_STABLE,
        "env_name": "academy_run_pass_and_shoot_with_keeper",
        "num_controlled_players": 2,
        "total_timesteps": 40_960,
        "save_interval": 10,
        "save_dir": "X_Jiang/checkpoints/academy_run_pass_and_shoot_attack_support",
        "logdir": "X_Jiang/logs/academy_run_pass_and_shoot_attack_support",
        "progressive_pass_choice_reward": 0.02,
        "progressive_pass_result_reward_scale": 0.04,
        "carry_progress_reward_scale": 0.04,
        "zone_entry_progress_reward": 0.06,
        "shot_choice_reward": 0.02,
        "shot_execution_reward": 0.07,
        "safe_reset_pass_reward": 0.0,
        "safe_reset_overuse_penalty": 0.03,
        "support_forward_lane_reward": 0.04,
        "support_behind_ball_penalty": 0.03,
        "second_player_support_reward": 0.015,
        "recover_shape_reward": 0.01,
        "hold_shape_reward": 0.005,
        "adaptive_progression_target": 0.35,
        "adaptive_on_ball_fraction_target": 0.18,
    },
    "academy_pass_and_shoot_attack_support": {
        **COMMON_ACADEMY_STABLE,
        "env_name": "academy_pass_and_shoot_with_keeper",
        "num_controlled_players": 2,
        "total_timesteps": 40_960,
        "save_interval": 10,
        "save_dir": "X_Jiang/checkpoints/academy_pass_and_shoot_attack_support",
        "logdir": "X_Jiang/logs/academy_pass_and_shoot_attack_support",
        "progressive_pass_choice_reward": 0.02,
        "progressive_pass_result_reward_scale": 0.05,
        "carry_progress_reward_scale": 0.02,
        "zone_entry_progress_reward": 0.04,
        "shot_choice_reward": 0.02,
        "shot_execution_reward": 0.08,
        "safe_reset_pass_reward": 0.0,
        "safe_reset_overuse_penalty": 0.04,
        "support_forward_lane_reward": 0.05,
        "support_behind_ball_penalty": 0.03,
        "second_player_support_reward": 0.015,
        "recover_shape_reward": 0.01,
        "hold_shape_reward": 0.005,
    },
    "academy_3v1_attack_support": {
        **COMMON_ACADEMY_STABLE,
        "env_name": "academy_3_vs_1_with_keeper",
        "num_controlled_players": 3,
        "total_timesteps": 61_440,
        "save_interval": 10,
        "save_dir": "X_Jiang/checkpoints/academy_3v1_attack_support",
        "logdir": "X_Jiang/logs/academy_3v1_attack_support",
        "progressive_pass_choice_reward": 0.02,
        "progressive_pass_result_reward_scale": 0.04,
        "carry_progress_reward_scale": 0.03,
        "zone_entry_progress_reward": 0.05,
        "shot_choice_reward": 0.02,
        "shot_execution_reward": 0.08,
        "safe_reset_pass_reward": 0.0,
        "safe_reset_overuse_penalty": 0.05,
        "support_forward_lane_reward": 0.04,
        "support_behind_ball_penalty": 0.03,
        "second_player_support_reward": 0.02,
        "recover_shape_reward": 0.015,
        "hold_shape_reward": 0.005,
        "adaptive_progression_target": 0.32,
        "adaptive_on_ball_fraction_target": 0.18,
    },
    "five_v_five_tactical_debug": {
        **COMMON_TACTICAL,
        "hidden_sizes": (128, 128),
        "total_timesteps": 25_600,
        "rollout_steps": 128,
        "update_epochs": 3,
        "num_minibatches": 4,
        "save_interval": 5,
        "log_interval": 1,
        "save_dir": "X_Jiang/checkpoints/five_v_five_tactical_debug",
        "logdir": "X_Jiang/logs/five_v_five_tactical_debug",
    },
    "five_v_five_tactical_30_updates": {
        **COMMON_TACTICAL,
        "hidden_sizes": (192, 192),
        "total_timesteps": 15_360,
        "rollout_steps": 128,
        "update_epochs": 4,
        "num_minibatches": 4,
        "save_interval": 10,
        "log_interval": 1,
        "save_dir": "X_Jiang/checkpoints/five_v_five_tactical_30_updates",
        "logdir": "X_Jiang/logs/five_v_five_tactical_30_updates",
    },
    "five_v_five_tactical_40_updates": {
        **COMMON_TACTICAL,
        "hidden_sizes": (192, 192),
        "total_timesteps": 25_600,
        "rollout_steps": 128,
        "update_epochs": 4,
        "num_minibatches": 4,
        "save_interval": 10,
        "log_interval": 1,
        "save_dir": "X_Jiang/checkpoints/five_v_five_tactical_40_updates",
        "logdir": "X_Jiang/logs/five_v_five_tactical_40_updates",
    },
    "five_v_five_tactical_50_updates": {
        **COMMON_TACTICAL,
        "hidden_sizes": (192, 192),
        "total_timesteps": 32_000,
        "rollout_steps": 128,
        "update_epochs": 4,
        "num_minibatches": 4,
        "save_interval": 10,
        "log_interval": 1,
        "save_dir": "X_Jiang/checkpoints/five_v_five_tactical_50_updates",
        "logdir": "X_Jiang/logs/five_v_five_tactical_50_updates",
    },
    "five_v_five_tactical_100_updates": {
        **COMMON_TACTICAL,
        "hidden_sizes": (192, 192),
        "total_timesteps": 64_000,
        "rollout_steps": 128,
        "update_epochs": 4,
        "num_minibatches": 4,
        "save_interval": 10,
        "log_interval": 1,
        "save_dir": "X_Jiang/checkpoints/five_v_five_tactical_100_updates",
        "logdir": "X_Jiang/logs/five_v_five_tactical_100_updates",
    },
    "five_v_five_tactical_200_updates": {
        **COMMON_TACTICAL,
        "hidden_sizes": (256, 256),
        "total_timesteps": 102_400,
        "rollout_steps": 128,
        "update_epochs": 4,
        "num_minibatches": 4,
        "save_interval": 20,
        "log_interval": 1,
        "save_dir": "X_Jiang/checkpoints/five_v_five_tactical_200_updates",
        "logdir": "X_Jiang/logs/five_v_five_tactical_200_updates",
    },
    "five_v_five_tactical_1000_updates": {
        **COMMON_TACTICAL,
        "hidden_sizes": (256, 256),
        "total_timesteps": 640_000,
        "rollout_steps": 128,
        "update_epochs": 4,
        "num_minibatches": 4,
        "save_interval": 50,
        "log_interval": 1,
        "save_dir": "X_Jiang/checkpoints/five_v_five_tactical_1000_updates",
        "logdir": "X_Jiang/logs/five_v_five_tactical_1000_updates",
    },
    "five_v_five_tactical_base": {
        **COMMON_TACTICAL,
        "total_timesteps": 400_000,
        "rollout_steps": 256,
        "update_epochs": 4,
        "num_minibatches": 4,
        "save_interval": 10,
        "log_interval": 1,
        "save_dir": "X_Jiang/checkpoints/five_v_five_tactical_base",
        "logdir": "X_Jiang/logs/five_v_five_tactical_base",
    },
}


def build_config(preset_name: str, overrides: dict[str, Any] | None = None) -> TrainConfig:
    if preset_name not in PRESETS:
        raise ValueError(f"Unknown preset: {preset_name}. Available presets: {list(PRESETS)}")
    merged: dict[str, Any] = dict(PRESETS[preset_name])
    if overrides:
        merged.update(overrides)
    return TrainConfig(**merged)
