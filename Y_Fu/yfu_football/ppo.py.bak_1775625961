from __future__ import annotations

import argparse
import math
import random
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from .envs import FootballVecEnv, RewardShapingConfig
from .model import MODEL_TYPES, ActorCritic

ACTION_NAMES = {
    0: "idle",
    1: "left",
    2: "top_left",
    3: "top",
    4: "top_right",
    5: "right",
    6: "bottom_right",
    7: "bottom",
    8: "bottom_left",
    9: "long_pass",
    10: "high_pass",
    11: "short_pass",
    12: "shot",
    13: "sprint",
    14: "release_direction",
    15: "release_sprint",
    16: "sliding",
    17: "dribble",
    18: "release_dribble",
}
DIRECTIONAL_ACTIONS = {1, 2, 3, 4, 5, 6, 7, 8}
RIGHTWARD_ACTIONS = {4, 5, 6}
LEFTWARD_ACTIONS = {1, 2, 8}
UPWARD_ACTIONS = {2, 3, 4}
DOWNWARD_ACTIONS = {6, 7, 8}
LONG_PASS_ACTIONS = {9}
HIGH_PASS_ACTIONS = {10}
SHORT_PASS_ACTIONS = {11}
PASS_ACTIONS = LONG_PASS_ACTIONS | HIGH_PASS_ACTIONS | SHORT_PASS_ACTIONS
SHOT_ACTIONS = {12}
BALL_SKILL_ACTIONS = {9, 10, 11, 12, 17}
IDLE_ACTIONS = {0}
RELEASE_DIRECTION_ACTIONS = {14}
SPRINT_ACTIONS = {13}
DRIBBLE_ACTIONS = {17}


@dataclass
class PPOConfig:
    env_name: str = "11_vs_11_easy_stochastic"
    representation: str = "simple115v2"
    rewards: str = "scoring,checkpoints"
    num_controlled_players: int = 11
    channel_dimensions: tuple[int, int] = (42, 42)
    model_type: str = "auto"
    feature_dim: int = 256
    use_player_id: bool = False
    total_timesteps: int = 500_000
    rollout_steps: int = 256
    num_envs: int = 1
    update_epochs: int = 4
    num_minibatches: int = 8
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    hidden_sizes: tuple[int, ...] = (256, 256)
    seed: int = 42
    render: bool = False
    save_interval: int = 10
    log_interval: int = 1
    save_dir: str = "Y_Fu/checkpoints"
    logdir: str = "Y_Fu/logs"
    device: str = "auto"
    init_checkpoint: str | None = None
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


PRESET_OVERRIDES: dict[str, dict[str, Any]] = {
    "default": {},
    "lightning": {
        "env_name": "academy_empty_goal_close",
        "representation": "extracted",
        "model_type": "cnn",
        "feature_dim": 256,
        "rewards": "scoring,checkpoints",
        "num_controlled_players": 1,
        "channel_dimensions": (42, 42),
        "total_timesteps": 20_000,
        "rollout_steps": 128,
        "update_epochs": 3,
        "num_minibatches": 4,
        "learning_rate": 3e-4,
        "hidden_sizes": (128, 128),
        "save_interval": 2,
        "save_dir": "Y_Fu/checkpoints/lightning",
        "logdir": "Y_Fu/logs/lightning",
    },
    "academy_run_to_score_with_keeper": {
        "env_name": "academy_run_to_score_with_keeper",
        "representation": "extracted",
        "model_type": "cnn",
        "feature_dim": 256,
        "rewards": "scoring,checkpoints",
        "num_controlled_players": 1,
        "channel_dimensions": (42, 42),
        "total_timesteps": 60_000,
        "rollout_steps": 128,
        "update_epochs": 3,
        "num_minibatches": 4,
        "learning_rate": 3e-4,
        "hidden_sizes": (128, 128),
        "save_interval": 5,
        "save_dir": "Y_Fu/checkpoints/academy_run_to_score_with_keeper",
        "logdir": "Y_Fu/logs/academy_run_to_score_with_keeper",
    },
    "academy_pass_and_shoot_with_keeper": {
        "env_name": "academy_pass_and_shoot_with_keeper",
        "representation": "extracted",
        "model_type": "cnn",
        "feature_dim": 256,
        "rewards": "scoring,checkpoints",
        "num_controlled_players": 2,
        "channel_dimensions": (42, 42),
        "total_timesteps": 100_000,
        "rollout_steps": 192,
        "update_epochs": 4,
        "num_minibatches": 4,
        "learning_rate": 3e-4,
        "hidden_sizes": (256, 256),
        "pass_success_reward": 0.03,
        "pass_failure_penalty": 0.03,
        "pass_progress_reward_scale": 0.00,
        "shot_attempt_reward": 0.30,
        "attacking_possession_reward": 0.00,
        "final_third_entry_reward": 0.05,
        "possession_retention_reward": 0.00,
        "possession_recovery_reward": 0.01,
        "defensive_third_recovery_reward": 0.01,
        "opponent_attacking_possession_penalty": 0.00,
        "own_half_turnover_penalty": 0.08,
        "own_half_x_threshold": 0.0,
        "save_interval": 5,
        "save_dir": "Y_Fu/checkpoints/academy_pass_and_shoot_with_keeper",
        "logdir": "Y_Fu/logs/academy_pass_and_shoot_with_keeper",
    },
    "academy_3_vs_1_with_keeper": {
        "env_name": "academy_3_vs_1_with_keeper",
        "representation": "extracted",
        "model_type": "cnn",
        "feature_dim": 256,
        "rewards": "scoring,checkpoints",
        "num_controlled_players": 3,
        "channel_dimensions": (42, 42),
        "total_timesteps": 150_000,
        "rollout_steps": 256,
        "update_epochs": 4,
        "num_minibatches": 4,
        "learning_rate": 3e-4,
        "hidden_sizes": (256, 256),
        "pass_success_reward": 0.08,
        "pass_failure_penalty": 0.05,
        "pass_progress_reward_scale": 0.08,
        "shot_attempt_reward": 0.03,
        "attacking_possession_reward": 0.002,
        "final_third_entry_reward": 0.04,
        "possession_retention_reward": 0.001,
        "possession_recovery_reward": 0.02,
        "defensive_third_recovery_reward": 0.03,
        "opponent_attacking_possession_penalty": 0.0015,
        "own_half_turnover_penalty": 0.02,
        "own_half_x_threshold": 0.0,
        "save_interval": 10,
        "save_dir": "Y_Fu/checkpoints/academy_3_vs_1_with_keeper",
        "logdir": "Y_Fu/logs/academy_3_vs_1_with_keeper",
    },
    "five_vs_five": {
        "env_name": "5_vs_5",
        "representation": "extracted",
        "model_type": "cnn",
        "feature_dim": 256,
        "rewards": "scoring,checkpoints",
        "num_controlled_players": 4,
        "channel_dimensions": (42, 42),
        "total_timesteps": 250_000,
        "rollout_steps": 512,
        "update_epochs": 4,
        "num_minibatches": 4,
        "learning_rate": 2.5e-4,
        "hidden_sizes": (256, 256),
        # Bias dense rewards toward creating and finishing attacks, not just
        # safe possession. This is the first reward-only revision after the
        # 10M-step five-v-five failure checkpoint.
        "pass_success_reward": 0.03,
        "pass_failure_penalty": 0.025,
        "pass_progress_reward_scale": 0.06,
        "shot_attempt_reward": 0.06,
        "attacking_possession_reward": 0.0,
        "final_third_entry_reward": 0.06,
        "possession_retention_reward": 0.0,
        "possession_recovery_reward": 0.015,
        "defensive_third_recovery_reward": 0.025,
        "opponent_attacking_possession_penalty": 0.0,
        "own_half_turnover_penalty": 0.015,
        "own_half_x_threshold": 0.0,
        "save_interval": 10,
        "save_dir": "Y_Fu/checkpoints/five_vs_five",
        "logdir": "Y_Fu/logs/five_vs_five",
    },
    "five_vs_five_reward_v2": {
        "env_name": "5_vs_5",
        "representation": "extracted",
        "model_type": "cnn",
        "feature_dim": 256,
        "rewards": "scoring,checkpoints",
        "num_controlled_players": 4,
        "channel_dimensions": (42, 42),
        "total_timesteps": 250_000,
        "rollout_steps": 512,
        "update_epochs": 4,
        "num_minibatches": 4,
        "learning_rate": 2.5e-4,
        "hidden_sizes": (256, 256),
        # Reward v2 is intentionally narrower and more football-process oriented.
        # It rewards getting into danger and converting danger into shots while
        # keeping the strongest clearly attributable transition penalty.
        "pass_success_reward": 0.02,
        "pass_failure_penalty": 0.02,
        "pass_progress_reward_scale": 0.02,
        "shot_attempt_reward": 0.08,
        "attacking_possession_reward": 0.0,
        "final_third_entry_reward": 0.08,
        "possession_retention_reward": 0.0,
        "possession_recovery_reward": 0.01,
        "defensive_third_recovery_reward": 0.015,
        "opponent_attacking_possession_penalty": 0.0,
        "own_half_turnover_penalty": 0.02,
        "own_half_x_threshold": 0.0,
        "save_interval": 10,
        "save_dir": "Y_Fu/checkpoints/five_vs_five_reward_v2",
        "logdir": "Y_Fu/logs/five_vs_five_reward_v2",
    },
    "five_vs_five_reward_v2b_transition": {
        "env_name": "5_vs_5",
        "representation": "extracted",
        "model_type": "cnn",
        "feature_dim": 256,
        "rewards": "scoring,checkpoints",
        "num_controlled_players": 4,
        "channel_dimensions": (42, 42),
        "total_timesteps": 250_000,
        "rollout_steps": 512,
        "update_epochs": 4,
        "num_minibatches": 4,
        "learning_rate": 2.5e-4,
        "hidden_sizes": (256, 256),
        # Transition-focused variant: slightly weaker pass incentives, slightly
        # stronger turnover discipline and defensive recovery.
        "pass_success_reward": 0.015,
        "pass_failure_penalty": 0.02,
        "pass_progress_reward_scale": 0.015,
        "shot_attempt_reward": 0.08,
        "attacking_possession_reward": 0.0,
        "final_third_entry_reward": 0.08,
        "possession_retention_reward": 0.0,
        "possession_recovery_reward": 0.01,
        "defensive_third_recovery_reward": 0.02,
        "opponent_attacking_possession_penalty": 0.0,
        "own_half_turnover_penalty": 0.03,
        "own_half_x_threshold": 0.0,
        "save_interval": 10,
        "save_dir": "Y_Fu/checkpoints/five_vs_five_reward_v2b_transition",
        "logdir": "Y_Fu/logs/five_vs_five_reward_v2b_transition",
    },
    "five_vs_five_reward_v2c_progression": {
        "env_name": "5_vs_5",
        "representation": "extracted",
        "model_type": "cnn",
        "feature_dim": 256,
        "rewards": "scoring,checkpoints",
        "num_controlled_players": 4,
        "channel_dimensions": (42, 42),
        "total_timesteps": 250_000,
        "rollout_steps": 512,
        "update_epochs": 4,
        "num_minibatches": 4,
        "learning_rate": 2.5e-4,
        "hidden_sizes": (256, 256),
        # Progression-focused variant: still avoids generic possession reward,
        # but tests whether line-breaking progress deserves slightly more credit.
        "pass_success_reward": 0.02,
        "pass_failure_penalty": 0.02,
        "pass_progress_reward_scale": 0.035,
        "shot_attempt_reward": 0.07,
        "attacking_possession_reward": 0.0,
        "final_third_entry_reward": 0.07,
        "possession_retention_reward": 0.0,
        "possession_recovery_reward": 0.01,
        "defensive_third_recovery_reward": 0.015,
        "opponent_attacking_possession_penalty": 0.0,
        "own_half_turnover_penalty": 0.02,
        "own_half_x_threshold": 0.0,
        "save_interval": 10,
        "save_dir": "Y_Fu/checkpoints/five_vs_five_reward_v2c_progression",
        "logdir": "Y_Fu/logs/five_vs_five_reward_v2c_progression",
    },
    "small_11v11": {
        "env_name": "11_vs_11_easy_stochastic",
        "representation": "simple115v2",
        "model_type": "residual_mlp",
        "feature_dim": 256,
        "rewards": "scoring,checkpoints",
        "num_controlled_players": 3,
        "channel_dimensions": (42, 42),
        "total_timesteps": 60_000,
        "rollout_steps": 128,
        "update_epochs": 3,
        "num_minibatches": 4,
        "hidden_sizes": (128, 128),
        "save_interval": 5,
        "save_dir": "Y_Fu/checkpoints/small_11v11",
        "logdir": "Y_Fu/logs/small_11v11",
    },
    "small_11v11_wide": {
        "env_name": "11_vs_11_easy_stochastic",
        "representation": "simple115v2",
        "model_type": "separate_mlp",
        "feature_dim": 256,
        "rewards": "scoring,checkpoints",
        "num_controlled_players": 3,
        "channel_dimensions": (42, 42),
        "total_timesteps": 120_000,
        "rollout_steps": 256,
        "update_epochs": 4,
        "num_minibatches": 4,
        "learning_rate": 3e-4,
        "hidden_sizes": (256, 256, 256),
        "save_interval": 10,
        "save_dir": "Y_Fu/checkpoints/small_11v11_wide",
        "logdir": "Y_Fu/logs/small_11v11_wide",
    },
    "full_11v11_residual": {
        "env_name": "11_vs_11_easy_stochastic",
        "representation": "simple115v2",
        "model_type": "residual_mlp",
        "feature_dim": 256,
        "rewards": "scoring,checkpoints",
        "num_controlled_players": 11,
        "channel_dimensions": (42, 42),
        "total_timesteps": 1_000_000,
        "rollout_steps": 512,
        "update_epochs": 4,
        "num_minibatches": 8,
        "learning_rate": 2.5e-4,
        "hidden_sizes": (512, 512, 512),
        "save_interval": 20,
        "save_dir": "Y_Fu/checkpoints/full_11v11_residual",
        "logdir": "Y_Fu/logs/full_11v11_residual",
    },
    "full_11v11_wide": {
        "env_name": "11_vs_11_easy_stochastic",
        "representation": "simple115v2",
        "model_type": "separate_mlp",
        "feature_dim": 256,
        "rewards": "scoring,checkpoints",
        "num_controlled_players": 11,
        "channel_dimensions": (42, 42),
        "total_timesteps": 1_000_000,
        "rollout_steps": 512,
        "update_epochs": 4,
        "num_minibatches": 8,
        "learning_rate": 2.5e-4,
        "hidden_sizes": (512, 512, 512),
        "save_interval": 20,
        "save_dir": "Y_Fu/checkpoints/full_11v11_wide",
        "logdir": "Y_Fu/logs/full_11v11_wide",
    },
}


def _resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _compute_gae(
    rewards: np.ndarray,
    values: np.ndarray,
    dones: np.ndarray,
    last_values: np.ndarray,
    gamma: float,
    gae_lambda: float,
) -> tuple[np.ndarray, np.ndarray]:
    advantages = np.zeros_like(rewards, dtype=np.float32)
    last_advantage = np.zeros_like(last_values, dtype=np.float32)

    for step in reversed(range(rewards.shape[0])):
        if step == rewards.shape[0] - 1:
            next_values = last_values
        else:
            next_values = values[step + 1]
        next_non_terminal = 1.0 - dones[step].astype(np.float32)
        while next_non_terminal.ndim < rewards[step].ndim:
            next_non_terminal = np.expand_dims(next_non_terminal, axis=-1)
        delta = rewards[step] + gamma * next_values * next_non_terminal - values[step]
        last_advantage = delta + gamma * gae_lambda * next_non_terminal * last_advantage
        advantages[step] = last_advantage

    returns = advantages + values
    return advantages, returns


def _safe_fraction(numerator: float, denominator: float) -> float:
    if denominator <= 0.0:
        return 0.0
    return float(numerator / denominator)


def _summarize_action_usage(action_buffer: np.ndarray, action_dim: int) -> dict[str, float | str]:
    flat_actions = action_buffer.reshape(-1)
    total_actions = int(flat_actions.size)
    if total_actions == 0:
        return {
            "pass_rate": 0.0,
            "shot_rate": 0.0,
            "skill_rate": 0.0,
            "idle_rate": 0.0,
            "direction_rate": 0.0,
            "right_bias": 0.0,
            "left_bias": 0.0,
            "vertical_bias": 0.0,
            "top_actions": "n/a",
        }

    counts = np.bincount(flat_actions, minlength=action_dim)
    directional_count = int(sum(counts[action] for action in DIRECTIONAL_ACTIONS if action < len(counts)))
    right_count = int(sum(counts[action] for action in RIGHTWARD_ACTIONS if action < len(counts)))
    left_count = int(sum(counts[action] for action in LEFTWARD_ACTIONS if action < len(counts)))
    up_count = int(sum(counts[action] for action in UPWARD_ACTIONS if action < len(counts)))
    down_count = int(sum(counts[action] for action in DOWNWARD_ACTIONS if action < len(counts)))
    long_pass_count = int(sum(counts[action] for action in LONG_PASS_ACTIONS if action < len(counts)))
    high_pass_count = int(sum(counts[action] for action in HIGH_PASS_ACTIONS if action < len(counts)))
    short_pass_count = int(sum(counts[action] for action in SHORT_PASS_ACTIONS if action < len(counts)))
    pass_count = int(sum(counts[action] for action in PASS_ACTIONS if action < len(counts)))
    shot_count = int(sum(counts[action] for action in SHOT_ACTIONS if action < len(counts)))
    skill_count = int(sum(counts[action] for action in BALL_SKILL_ACTIONS if action < len(counts)))
    idle_count = int(sum(counts[action] for action in IDLE_ACTIONS if action < len(counts)))
    release_direction_count = int(sum(counts[action] for action in RELEASE_DIRECTION_ACTIONS if action < len(counts)))
    sprint_count = int(sum(counts[action] for action in SPRINT_ACTIONS if action < len(counts)))
    dribble_count = int(sum(counts[action] for action in DRIBBLE_ACTIONS if action < len(counts)))

    directional_counts = np.asarray(
        [counts[action] for action in sorted(DIRECTIONAL_ACTIONS) if action < len(counts)],
        dtype=np.float32,
    )
    if directional_count > 0:
        directional_probs = directional_counts / float(directional_count)
        direction_entropy = float(-np.sum(np.where(directional_probs > 0.0, directional_probs * np.log(directional_probs), 0.0)))
        normalized_direction_entropy = direction_entropy / math.log(len(directional_probs))
    else:
        direction_entropy = 0.0
        normalized_direction_entropy = 0.0

    ranked_actions = sorted(
        ((action_id, int(count)) for action_id, count in enumerate(counts)),
        key=lambda item: item[1],
        reverse=True,
    )
    top_actions = []
    for action_id, count in ranked_actions[:3]:
        if count <= 0:
            continue
        action_name = ACTION_NAMES.get(action_id, str(action_id))
        top_actions.append(f"{action_name}:{_safe_fraction(count, total_actions):.2f}")

    return {
        "pass_rate": _safe_fraction(pass_count, total_actions),
        "long_pass_rate": _safe_fraction(long_pass_count, total_actions),
        "high_pass_rate": _safe_fraction(high_pass_count, total_actions),
        "short_pass_rate": _safe_fraction(short_pass_count, total_actions),
        "shot_rate": _safe_fraction(shot_count, total_actions),
        "skill_rate": _safe_fraction(skill_count, total_actions),
        "idle_rate": _safe_fraction(idle_count, total_actions),
        "release_direction_rate": _safe_fraction(release_direction_count, total_actions),
        "sprint_rate": _safe_fraction(sprint_count, total_actions),
        "dribble_rate": _safe_fraction(dribble_count, total_actions),
        "direction_rate": _safe_fraction(directional_count, total_actions),
        "right_bias": _safe_fraction(right_count, directional_count),
        "left_bias": _safe_fraction(left_count, directional_count),
        "up_bias": _safe_fraction(up_count, directional_count),
        "down_bias": _safe_fraction(down_count, directional_count),
        "non_right_direction_rate": _safe_fraction(directional_count - right_count, directional_count),
        "dominant_action_share": _safe_fraction(ranked_actions[0][1], total_actions) if ranked_actions else 0.0,
        "direction_entropy": direction_entropy,
        "direction_entropy_norm": normalized_direction_entropy,
        "vertical_bias": _safe_fraction(abs(up_count - down_count), directional_count),
        "top_actions": ",".join(top_actions) if top_actions else "n/a",
    }


def _extract_executed_actions(
    infos: Sequence[dict[str, Any]],
    sampled_actions: np.ndarray,
    num_players: int,
) -> tuple[np.ndarray, dict[str, float]]:
    executed_actions = np.asarray(sampled_actions, dtype=np.int64).copy()
    invalid_ball_skill_count = 0.0
    invalid_no_ball_pass_count = 0.0
    invalid_no_ball_shot_count = 0.0
    invalid_no_ball_dribble_count = 0.0

    for env_index, info in enumerate(infos):
        info_dict = dict(info or {})
        env_executed_actions = info_dict.get("executed_actions")
        if env_executed_actions is not None:
            executed_actions[env_index] = np.asarray(env_executed_actions, dtype=np.int64).reshape(num_players)
        invalid_ball_skill_count += float(info_dict.get("invalid_ball_skill_count", 0.0))
        invalid_no_ball_pass_count += float(info_dict.get("invalid_no_ball_pass_count", 0.0))
        invalid_no_ball_shot_count += float(info_dict.get("invalid_no_ball_shot_count", 0.0))
        invalid_no_ball_dribble_count += float(info_dict.get("invalid_no_ball_dribble_count", 0.0))

    return executed_actions, {
        "invalid_ball_skill_count": invalid_ball_skill_count,
        "invalid_no_ball_pass_count": invalid_no_ball_pass_count,
        "invalid_no_ball_shot_count": invalid_no_ball_shot_count,
        "invalid_no_ball_dribble_count": invalid_no_ball_dribble_count,
    }


def _load_compatible_module_state(target_module: nn.Module, source_state: dict[str, torch.Tensor]) -> tuple[int, int]:
    target_state = target_module.state_dict()
    compatible = {}
    skipped = 0
    for key, value in source_state.items():
        if key in target_state and target_state[key].shape == value.shape:
            compatible[key] = value
        else:
            skipped += 1
    target_module.load_state_dict(compatible, strict=False)
    return len(compatible), skipped


class PPOTrainer:
    def __init__(self, config: PPOConfig) -> None:
        self.config = config
        self.device = _resolve_device(config.device)
        _set_seed(config.seed)

        self.env = FootballVecEnv(
            num_envs=config.num_envs,
            base_seed=config.seed,
            env_name=config.env_name,
            representation=config.representation,
            rewards=config.rewards,
            render=config.render,
            logdir=config.logdir,
            num_controlled_players=config.num_controlled_players,
            channel_dimensions=config.channel_dimensions,
            reward_shaping=RewardShapingConfig(
                pass_success_reward=config.pass_success_reward,
                pass_failure_penalty=config.pass_failure_penalty,
                pass_progress_reward_scale=config.pass_progress_reward_scale,
                shot_attempt_reward=config.shot_attempt_reward,
                attacking_possession_reward=config.attacking_possession_reward,
                attacking_x_threshold=config.attacking_x_threshold,
                final_third_entry_reward=config.final_third_entry_reward,
                possession_retention_reward=config.possession_retention_reward,
                possession_recovery_reward=config.possession_recovery_reward,
                defensive_third_recovery_reward=config.defensive_third_recovery_reward,
                opponent_attacking_possession_penalty=config.opponent_attacking_possession_penalty,
                own_half_turnover_penalty=config.own_half_turnover_penalty,
                own_half_x_threshold=config.own_half_x_threshold,
                defensive_x_threshold=config.defensive_x_threshold,
                pending_pass_horizon=config.pending_pass_horizon,
            ),
        )
        self.model = ActorCritic(
            obs_dim=self.env.obs_dim,
            action_dim=self.env.action_dim,
            hidden_sizes=config.hidden_sizes,
            obs_shape=self.env.obs_shape,
            model_type=config.model_type,
            feature_dim=config.feature_dim,
            player_id_dim=self.env.num_players if config.use_player_id else 0,
        ).to(self.device)
        self.init_checkpoint = config.init_checkpoint
        if self.init_checkpoint:
            self._load_initial_checkpoint(self.init_checkpoint)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.learning_rate, eps=1e-5)
        self.save_dir = Path(config.save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        Path(config.logdir).mkdir(parents=True, exist_ok=True)
        if self.init_checkpoint:
            print(f"Initialized model from checkpoint: {self.init_checkpoint}")

        self.total_env_steps = 0
        self.total_agent_steps = 0
        self.current_episode_return = np.zeros(self.env.num_envs, dtype=np.float32)
        self.current_episode_score_reward = np.zeros(self.env.num_envs, dtype=np.float32)
        self.current_episode_length = np.zeros(self.env.num_envs, dtype=np.int32)
        self.current_episode_pass_attempts = np.zeros(self.env.num_envs, dtype=np.float32)
        self.current_episode_pass_successes = np.zeros(self.env.num_envs, dtype=np.float32)
        self.current_episode_pass_progress = np.zeros(self.env.num_envs, dtype=np.float32)
        self.current_episode_shot_attempts = np.zeros(self.env.num_envs, dtype=np.float32)
        self.current_episode_final_third_entries = np.zeros(self.env.num_envs, dtype=np.float32)
        self.current_episode_own_half_turnovers = np.zeros(self.env.num_envs, dtype=np.float32)
        self.current_episode_possession_recoveries = np.zeros(self.env.num_envs, dtype=np.float32)
        self.current_episode_defensive_third_recoveries = np.zeros(self.env.num_envs, dtype=np.float32)
        self.current_episode_opponent_dangerous_possessions = np.zeros(self.env.num_envs, dtype=np.float32)
        self.player_id_matrix = np.broadcast_to(
            np.arange(self.env.num_players, dtype=np.int64),
            (self.env.num_envs, self.env.num_players),
        ).copy()

    def _load_initial_checkpoint(self, checkpoint_path: str) -> None:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
            try:
                self.model.load_state_dict(state_dict, strict=True)
            except RuntimeError:
                loaded, skipped = _load_compatible_module_state(self.model, state_dict)
                print(
                    "Initialized PPO from partially compatible checkpoint: "
                    f"{loaded} tensors loaded, {skipped} skipped."
                )
            return
        if "q1_state_dict" in checkpoint and "q2_state_dict" in checkpoint and "v_state_dict" in checkpoint:
            from .iql import DiscreteIQL

            if self.model.model_type == "separate_mlp":
                raise ValueError("Initializing PPO from an IQL checkpoint is not supported for model_type='separate_mlp'.")

            iql = DiscreteIQL.load_checkpoint(checkpoint_path, device="cpu")
            encoder_loaded, encoder_skipped = _load_compatible_module_state(
                self.model.encoder,
                iql.q1.encoder.state_dict(),
            )
            actor_loaded, actor_skipped = _load_compatible_module_state(
                self.model.actor_body,
                iql.q1.body.state_dict(),
            )
            policy_loaded, policy_skipped = _load_compatible_module_state(
                self.model.policy_head,
                iql.q1.output_head.state_dict(),
            )
            critic_loaded, critic_skipped = _load_compatible_module_state(
                self.model.critic_body,
                iql.v.body.state_dict(),
            )
            value_loaded, value_skipped = _load_compatible_module_state(
                self.model.value_head,
                iql.v.output_head.state_dict(),
            )
            print(
                "Initialized PPO from IQL checkpoint: "
                f"encoder={encoder_loaded} loaded/{encoder_skipped} skipped, "
                f"actor_body={actor_loaded}/{actor_skipped}, "
                f"policy_head={policy_loaded}/{policy_skipped}, "
                f"critic_body={critic_loaded}/{critic_skipped}, "
                f"value_head={value_loaded}/{value_skipped}."
            )
            return
        raise ValueError(
            f"Unsupported init checkpoint format: {checkpoint_path}. "
            "Expected PPO 'model_state_dict' or IQL 'q1_state_dict'/'q2_state_dict'/'v_state_dict'."
        )

    def collect_rollout(self, observation: np.ndarray) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, float]]:
        steps = self.config.rollout_steps
        num_envs = self.env.num_envs
        num_players = self.env.num_players

        obs_buffer = np.zeros((steps, num_envs, num_players, self.env.obs_dim), dtype=np.float32)
        action_buffer = np.zeros((steps, num_envs, num_players), dtype=np.int64)
        logprob_buffer = np.zeros((steps, num_envs, num_players), dtype=np.float32)
        reward_buffer = np.zeros((steps, num_envs, num_players), dtype=np.float32)
        done_buffer = np.zeros((steps, num_envs), dtype=np.bool_)
        value_buffer = np.zeros((steps, num_envs, num_players), dtype=np.float32)

        completed_returns: list[float] = []
        completed_lengths: list[int] = []
        completed_score_rewards: list[float] = []
        completed_successes: list[float] = []
        completed_scores: list[tuple[int, int]] = []
        completed_pass_attempts: list[float] = []
        completed_pass_successes: list[float] = []
        completed_pass_progress: list[float] = []
        completed_shot_attempts: list[float] = []
        completed_final_third_entries: list[float] = []
        completed_own_half_turnovers: list[float] = []
        completed_possession_recoveries: list[float] = []
        completed_defensive_third_recoveries: list[float] = []
        completed_opponent_dangerous_possessions: list[float] = []
        invalid_ball_skill_count_total = 0.0
        invalid_no_ball_pass_count_total = 0.0
        invalid_no_ball_shot_count_total = 0.0
        invalid_no_ball_dribble_count_total = 0.0

        for step in range(steps):
            obs_buffer[step] = observation
            flat_observation = observation.reshape(num_envs * num_players, self.env.obs_dim)
            obs_tensor = torch.as_tensor(flat_observation, dtype=torch.float32, device=self.device)
            player_ids_tensor = torch.as_tensor(
                self.player_id_matrix.reshape(num_envs * num_players),
                dtype=torch.int64,
                device=self.device,
            )
            with torch.no_grad():
                actions, _, values = self.model.act(obs_tensor, player_ids=player_ids_tensor)

            sampled_action_np = actions.cpu().numpy().reshape(num_envs, num_players)
            next_observation, reward, done, infos = self.env.step(sampled_action_np)
            executed_action_np, invalid_action_counts = _extract_executed_actions(
                infos,
                sampled_action_np,
                num_players,
            )
            executed_action_tensor = torch.as_tensor(
                executed_action_np.reshape(num_envs * num_players),
                dtype=torch.int64,
                device=self.device,
            )
            with torch.no_grad():
                _, executed_logprobs, _, _ = self.model.get_action_and_value(
                    obs_tensor,
                    player_ids=player_ids_tensor,
                    action=executed_action_tensor,
                )

            action_buffer[step] = executed_action_np
            logprob_buffer[step] = executed_logprobs.cpu().numpy().reshape(num_envs, num_players)
            value_buffer[step] = values.cpu().numpy().reshape(num_envs, num_players)
            reward_buffer[step] = reward
            done_buffer[step] = done
            invalid_ball_skill_count_total += invalid_action_counts["invalid_ball_skill_count"]
            invalid_no_ball_pass_count_total += invalid_action_counts["invalid_no_ball_pass_count"]
            invalid_no_ball_shot_count_total += invalid_action_counts["invalid_no_ball_shot_count"]
            invalid_no_ball_dribble_count_total += invalid_action_counts["invalid_no_ball_dribble_count"]

            self.current_episode_return += reward.mean(axis=1)
            self.current_episode_score_reward += np.asarray(
                [float(info.get("score_reward", 0.0)) for info in infos],
                dtype=np.float32,
            )
            self.current_episode_pass_attempts += np.asarray(
                [float(info.get("pass_attempt_event", 0.0)) for info in infos],
                dtype=np.float32,
            )
            self.current_episode_pass_successes += np.asarray(
                [float(info.get("pass_success_event", 0.0)) for info in infos],
                dtype=np.float32,
            )
            self.current_episode_pass_progress += np.asarray(
                [float(info.get("pass_progress_delta", 0.0)) for info in infos],
                dtype=np.float32,
            )
            self.current_episode_shot_attempts += np.asarray(
                [float(info.get("shot_attempt_event", 0.0)) for info in infos],
                dtype=np.float32,
            )
            self.current_episode_final_third_entries += np.asarray(
                [float(info.get("final_third_entry_event", 0.0)) for info in infos],
                dtype=np.float32,
            )
            self.current_episode_own_half_turnovers += np.asarray(
                [float(info.get("own_half_turnover_event", 0.0)) for info in infos],
                dtype=np.float32,
            )
            self.current_episode_possession_recoveries += np.asarray(
                [float(info.get("possession_recovery_event", 0.0)) for info in infos],
                dtype=np.float32,
            )
            self.current_episode_defensive_third_recoveries += np.asarray(
                [float(info.get("defensive_third_recovery_event", 0.0)) for info in infos],
                dtype=np.float32,
            )
            self.current_episode_opponent_dangerous_possessions += np.asarray(
                [float(info.get("opponent_dangerous_possession_event", 0.0)) for info in infos],
                dtype=np.float32,
            )
            self.current_episode_length += 1
            self.total_env_steps += num_envs
            self.total_agent_steps += num_envs * num_players

            observation = next_observation

            for env_index, env_done in enumerate(done):
                if not env_done:
                    continue

                info = infos[env_index]
                score = info.get("final_score")
                if score is not None:
                    success = 1.0 if score[0] > score[1] else 0.0
                    completed_scores.append((int(score[0]), int(score[1])))
                else:
                    success = 1.0 if self.current_episode_score_reward[env_index] > 0.0 else 0.0
                completed_returns.append(float(self.current_episode_return[env_index]))
                completed_lengths.append(int(self.current_episode_length[env_index]))
                completed_score_rewards.append(float(self.current_episode_score_reward[env_index]))
                completed_successes.append(success)
                completed_pass_attempts.append(float(self.current_episode_pass_attempts[env_index]))
                completed_pass_successes.append(float(self.current_episode_pass_successes[env_index]))
                completed_pass_progress.append(float(self.current_episode_pass_progress[env_index]))
                completed_shot_attempts.append(float(self.current_episode_shot_attempts[env_index]))
                completed_final_third_entries.append(float(self.current_episode_final_third_entries[env_index]))
                completed_own_half_turnovers.append(float(self.current_episode_own_half_turnovers[env_index]))
                completed_possession_recoveries.append(float(self.current_episode_possession_recoveries[env_index]))
                completed_defensive_third_recoveries.append(float(self.current_episode_defensive_third_recoveries[env_index]))
                completed_opponent_dangerous_possessions.append(
                    float(self.current_episode_opponent_dangerous_possessions[env_index])
                )
                self.current_episode_return[env_index] = 0.0
                self.current_episode_score_reward[env_index] = 0.0
                self.current_episode_length[env_index] = 0
                self.current_episode_pass_attempts[env_index] = 0.0
                self.current_episode_pass_successes[env_index] = 0.0
                self.current_episode_pass_progress[env_index] = 0.0
                self.current_episode_shot_attempts[env_index] = 0.0
                self.current_episode_final_third_entries[env_index] = 0.0
                self.current_episode_own_half_turnovers[env_index] = 0.0
                self.current_episode_possession_recoveries[env_index] = 0.0
                self.current_episode_defensive_third_recoveries[env_index] = 0.0
                self.current_episode_opponent_dangerous_possessions[env_index] = 0.0

        with torch.no_grad():
            last_values = self.model.get_value(
                torch.as_tensor(
                    observation.reshape(num_envs * num_players, self.env.obs_dim),
                    dtype=torch.float32,
                    device=self.device,
                ),
                player_ids=torch.as_tensor(
                    self.player_id_matrix.reshape(num_envs * num_players),
                    dtype=torch.int64,
                    device=self.device,
                ),
            ).cpu().numpy().reshape(num_envs, num_players)

        advantages, returns = _compute_gae(
            rewards=reward_buffer,
            values=value_buffer,
            dones=done_buffer,
            last_values=last_values,
            gamma=self.config.gamma,
            gae_lambda=self.config.gae_lambda,
        )

        batch = {
            "obs": obs_buffer.reshape(-1, self.env.obs_dim),
            "player_ids": np.broadcast_to(
                np.arange(self.env.num_players, dtype=np.int64),
                (steps, num_envs, num_players),
            ).reshape(-1),
            "actions": action_buffer.reshape(-1),
            "logprobs": logprob_buffer.reshape(-1),
            "advantages": advantages.reshape(-1),
            "returns": returns.reshape(-1),
            "values": value_buffer.reshape(-1),
        }
        score_examples = (
            ",".join(f"{left}-{right}" for left, right in completed_scores[-3:])
            if completed_scores
            else "n/a"
        )
        action_usage = _summarize_action_usage(action_buffer, self.env.action_dim)
        total_actions = float(action_buffer.size)
        total_completed_pass_attempts = float(np.sum(completed_pass_attempts)) if completed_pass_attempts else 0.0
        total_completed_pass_successes = float(np.sum(completed_pass_successes)) if completed_pass_successes else 0.0
        total_completed_final_third_entries = float(np.sum(completed_final_third_entries)) if completed_final_third_entries else 0.0
        total_completed_shot_attempts = float(np.sum(completed_shot_attempts)) if completed_shot_attempts else 0.0
        metrics = {
            "episodes_finished": float(len(completed_returns)),
            "mean_episode_return": float(np.mean(completed_returns)) if completed_returns else float("nan"),
            "mean_episode_length": float(np.mean(completed_lengths)) if completed_lengths else float("nan"),
            "min_episode_length": float(np.min(completed_lengths)) if completed_lengths else float("nan"),
            "max_episode_length": float(np.max(completed_lengths)) if completed_lengths else float("nan"),
            "mean_score_reward": float(np.mean(completed_score_rewards)) if completed_score_rewards else float("nan"),
            "mean_goals_for": float(np.mean([left for left, _ in completed_scores])) if completed_scores else float("nan"),
            "mean_goals_against": float(np.mean([right for _, right in completed_scores])) if completed_scores else float("nan"),
            "score_examples": score_examples,
            "success_rate": float(np.mean(completed_successes)) if completed_successes else float("nan"),
            "mean_pass_attempts": float(np.mean(completed_pass_attempts)) if completed_pass_attempts else float("nan"),
            "mean_pass_successes": float(np.mean(completed_pass_successes)) if completed_pass_successes else float("nan"),
            "pass_success_per_attempt": _safe_fraction(total_completed_pass_successes, total_completed_pass_attempts),
            "mean_pass_progress": float(np.mean(completed_pass_progress)) if completed_pass_progress else float("nan"),
            "mean_shot_attempt_events": float(np.mean(completed_shot_attempts)) if completed_shot_attempts else float("nan"),
            "mean_final_third_entries": float(np.mean(completed_final_third_entries)) if completed_final_third_entries else float("nan"),
            "shot_per_final_third_entry": _safe_fraction(
                total_completed_shot_attempts,
                total_completed_final_third_entries,
            ),
            "mean_own_half_turnovers": float(np.mean(completed_own_half_turnovers)) if completed_own_half_turnovers else float("nan"),
            "mean_possession_recoveries": float(np.mean(completed_possession_recoveries)) if completed_possession_recoveries else float("nan"),
            "mean_defensive_third_recoveries": float(np.mean(completed_defensive_third_recoveries)) if completed_defensive_third_recoveries else float("nan"),
            "mean_opponent_dangerous_possessions": float(np.mean(completed_opponent_dangerous_possessions))
            if completed_opponent_dangerous_possessions
            else float("nan"),
            "pass_rate": float(action_usage["pass_rate"]),
            "long_pass_rate": float(action_usage["long_pass_rate"]),
            "high_pass_rate": float(action_usage["high_pass_rate"]),
            "short_pass_rate": float(action_usage["short_pass_rate"]),
            "shot_rate": float(action_usage["shot_rate"]),
            "skill_rate": float(action_usage["skill_rate"]),
            "idle_rate": float(action_usage["idle_rate"]),
            "release_direction_rate": float(action_usage["release_direction_rate"]),
            "sprint_rate": float(action_usage["sprint_rate"]),
            "dribble_rate": float(action_usage["dribble_rate"]),
            "direction_rate": float(action_usage["direction_rate"]),
            "right_bias": float(action_usage["right_bias"]),
            "left_bias": float(action_usage["left_bias"]),
            "up_bias": float(action_usage["up_bias"]),
            "down_bias": float(action_usage["down_bias"]),
            "non_right_direction_rate": float(action_usage["non_right_direction_rate"]),
            "dominant_action_share": float(action_usage["dominant_action_share"]),
            "direction_entropy": float(action_usage["direction_entropy"]),
            "direction_entropy_norm": float(action_usage["direction_entropy_norm"]),
            "vertical_bias": float(action_usage["vertical_bias"]),
            "invalid_ball_skill_rate": _safe_fraction(invalid_ball_skill_count_total, total_actions),
            "invalid_no_ball_pass_rate": _safe_fraction(invalid_no_ball_pass_count_total, total_actions),
            "invalid_no_ball_shot_rate": _safe_fraction(invalid_no_ball_shot_count_total, total_actions),
            "invalid_no_ball_dribble_rate": _safe_fraction(invalid_no_ball_dribble_count_total, total_actions),
            "top_actions": str(action_usage["top_actions"]),
        }
        return observation, batch, metrics

    def update(self, batch: dict[str, np.ndarray]) -> dict[str, float]:
        obs = torch.as_tensor(batch["obs"], dtype=torch.float32, device=self.device)
        player_ids = torch.as_tensor(batch["player_ids"], dtype=torch.int64, device=self.device)
        actions = torch.as_tensor(batch["actions"], dtype=torch.int64, device=self.device)
        old_logprobs = torch.as_tensor(batch["logprobs"], dtype=torch.float32, device=self.device)
        advantages = torch.as_tensor(batch["advantages"], dtype=torch.float32, device=self.device)
        returns = torch.as_tensor(batch["returns"], dtype=torch.float32, device=self.device)
        old_values = torch.as_tensor(batch["values"], dtype=torch.float32, device=self.device)

        if advantages.numel() > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)

        batch_size = obs.shape[0]
        minibatch_size = batch_size // self.config.num_minibatches
        if minibatch_size < 1:
            raise ValueError("num_minibatches is too large for the collected rollout.")

        policy_losses: list[float] = []
        value_losses: list[float] = []
        entropies: list[float] = []
        approx_kls: list[float] = []
        clipfracs: list[float] = []

        for _ in range(self.config.update_epochs):
            indices = np.random.permutation(batch_size)
            for start in range(0, batch_size, minibatch_size):
                minibatch_indices = indices[start : start + minibatch_size]
                _, new_logprob, entropy, new_value = self.model.get_action_and_value(
                    obs[minibatch_indices],
                    player_ids[minibatch_indices],
                    actions[minibatch_indices],
                )
                logratio = new_logprob - old_logprobs[minibatch_indices]
                ratio = logratio.exp()

                with torch.no_grad():
                    approx_kl = ((ratio - 1.0) - logratio).mean()
                    clipfracs.append(
                        float(((ratio - 1.0).abs() > self.config.clip_coef).float().mean().item())
                    )

                minibatch_advantages = advantages[minibatch_indices]
                policy_loss_1 = -minibatch_advantages * ratio
                policy_loss_2 = -minibatch_advantages * torch.clamp(
                    ratio,
                    1.0 - self.config.clip_coef,
                    1.0 + self.config.clip_coef,
                )
                policy_loss = torch.max(policy_loss_1, policy_loss_2).mean()

                value_delta = new_value - old_values[minibatch_indices]
                value_clipped = old_values[minibatch_indices] + torch.clamp(
                    value_delta,
                    -self.config.clip_coef,
                    self.config.clip_coef,
                )
                value_loss_unclipped = (new_value - returns[minibatch_indices]) ** 2
                value_loss_clipped = (value_clipped - returns[minibatch_indices]) ** 2
                value_loss = 0.5 * torch.max(value_loss_unclipped, value_loss_clipped).mean()

                entropy_loss = entropy.mean()
                loss = policy_loss + self.config.vf_coef * value_loss - self.config.ent_coef * entropy_loss

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                self.optimizer.step()

                policy_losses.append(float(policy_loss.item()))
                value_losses.append(float(value_loss.item()))
                entropies.append(float(entropy_loss.item()))
                approx_kls.append(float(approx_kl.item()))

        return {
            "policy_loss": float(np.mean(policy_losses)),
            "value_loss": float(np.mean(value_losses)),
            "entropy": float(np.mean(entropies)),
            "approx_kl": float(np.mean(approx_kls)),
            "clipfrac": float(np.mean(clipfracs)) if clipfracs else 0.0,
        }

    def save_checkpoint(self, name: str, update: int) -> Path:
        checkpoint_path = self.save_dir / name
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "config": asdict(self.config),
                "obs_dim": self.env.obs_dim,
                "obs_shape": self.env.obs_shape,
                "action_dim": self.env.action_dim,
                "num_envs": self.env.num_envs,
                "num_players": self.env.num_players,
                "update": update,
                "total_env_steps": self.total_env_steps,
                "total_agent_steps": self.total_agent_steps,
            },
            checkpoint_path,
        )
        return checkpoint_path

    def train(self) -> Path:
        env_steps_per_update = self.config.rollout_steps * self.env.num_envs
        agent_steps_per_update = env_steps_per_update * self.env.num_players
        steps_per_update = agent_steps_per_update
        num_updates = math.ceil(self.config.total_timesteps / steps_per_update)
        observation = self.env.reset()
        latest_checkpoint = self.save_dir / "latest.pt"
        train_start = time.perf_counter()

        try:
            for update in range(1, num_updates + 1):
                rollout_start = time.perf_counter()
                observation, batch, rollout_metrics = self.collect_rollout(observation)
                rollout_time = max(time.perf_counter() - rollout_start, 1e-8)

                update_start = time.perf_counter()
                update_metrics = self.update(batch)
                update_time = max(time.perf_counter() - update_start, 1e-8)
                total_elapsed = max(time.perf_counter() - train_start, 1e-8)

                if update % self.config.log_interval == 0:
                    return_text = (
                        f"{rollout_metrics['mean_episode_return']:.3f}"
                        if math.isfinite(rollout_metrics["mean_episode_return"])
                        else "n/a"
                    )
                    length_text = (
                        f"{rollout_metrics['mean_episode_length']:.1f}"
                        if math.isfinite(rollout_metrics["mean_episode_length"])
                        else "n/a"
                    )
                    length_range_text = (
                        f"{int(rollout_metrics['min_episode_length'])}-{int(rollout_metrics['max_episode_length'])}"
                        if math.isfinite(rollout_metrics["min_episode_length"])
                        and math.isfinite(rollout_metrics["max_episode_length"])
                        else "n/a"
                    )
                    score_reward_text = (
                        f"{rollout_metrics['mean_score_reward']:.3f}"
                        if math.isfinite(rollout_metrics["mean_score_reward"])
                        else "n/a"
                    )
                    mean_pass_attempts_text = (
                        f"{rollout_metrics['mean_pass_attempts']:.2f}"
                        if math.isfinite(rollout_metrics["mean_pass_attempts"])
                        else "n/a"
                    )
                    mean_pass_successes_text = (
                        f"{rollout_metrics['mean_pass_successes']:.2f}"
                        if math.isfinite(rollout_metrics["mean_pass_successes"])
                        else "n/a"
                    )
                    pass_success_per_attempt_text = f"{rollout_metrics['pass_success_per_attempt']:.2f}"
                    mean_pass_progress_text = (
                        f"{rollout_metrics['mean_pass_progress']:.2f}"
                        if math.isfinite(rollout_metrics["mean_pass_progress"])
                        else "n/a"
                    )
                    mean_shot_attempt_events_text = (
                        f"{rollout_metrics['mean_shot_attempt_events']:.2f}"
                        if math.isfinite(rollout_metrics["mean_shot_attempt_events"])
                        else "n/a"
                    )
                    mean_final_third_entries_text = (
                        f"{rollout_metrics['mean_final_third_entries']:.2f}"
                        if math.isfinite(rollout_metrics["mean_final_third_entries"])
                        else "n/a"
                    )
                    shot_per_final_third_entry_text = f"{rollout_metrics['shot_per_final_third_entry']:.2f}"
                    mean_own_half_turnovers_text = (
                        f"{rollout_metrics['mean_own_half_turnovers']:.2f}"
                        if math.isfinite(rollout_metrics["mean_own_half_turnovers"])
                        else "n/a"
                    )
                    mean_possession_recoveries_text = (
                        f"{rollout_metrics['mean_possession_recoveries']:.2f}"
                        if math.isfinite(rollout_metrics["mean_possession_recoveries"])
                        else "n/a"
                    )
                    mean_defensive_third_recoveries_text = (
                        f"{rollout_metrics['mean_defensive_third_recoveries']:.2f}"
                        if math.isfinite(rollout_metrics["mean_defensive_third_recoveries"])
                        else "n/a"
                    )
                    mean_opponent_dangerous_possessions_text = (
                        f"{rollout_metrics['mean_opponent_dangerous_possessions']:.2f}"
                        if math.isfinite(rollout_metrics["mean_opponent_dangerous_possessions"])
                        else "n/a"
                    )
                    success_text = (
                        f"{rollout_metrics['success_rate']:.3f}"
                        if math.isfinite(rollout_metrics["success_rate"])
                        else "n/a"
                    )
                    goals_for_text = (
                        f"{rollout_metrics['mean_goals_for']:.2f}"
                        if math.isfinite(rollout_metrics["mean_goals_for"])
                        else "n/a"
                    )
                    goals_against_text = (
                        f"{rollout_metrics['mean_goals_against']:.2f}"
                        if math.isfinite(rollout_metrics["mean_goals_against"])
                        else "n/a"
                    )
                    pass_rate_text = f"{rollout_metrics['pass_rate']:.2f}"
                    long_pass_rate_text = f"{rollout_metrics['long_pass_rate']:.2f}"
                    high_pass_rate_text = f"{rollout_metrics['high_pass_rate']:.2f}"
                    short_pass_rate_text = f"{rollout_metrics['short_pass_rate']:.2f}"
                    shot_rate_text = f"{rollout_metrics['shot_rate']:.2f}"
                    skill_rate_text = f"{rollout_metrics['skill_rate']:.2f}"
                    idle_rate_text = f"{rollout_metrics['idle_rate']:.2f}"
                    release_direction_rate_text = f"{rollout_metrics['release_direction_rate']:.2f}"
                    sprint_rate_text = f"{rollout_metrics['sprint_rate']:.2f}"
                    dribble_rate_text = f"{rollout_metrics['dribble_rate']:.2f}"
                    direction_rate_text = f"{rollout_metrics['direction_rate']:.2f}"
                    right_bias_text = f"{rollout_metrics['right_bias']:.2f}"
                    left_bias_text = f"{rollout_metrics['left_bias']:.2f}"
                    up_bias_text = f"{rollout_metrics['up_bias']:.2f}"
                    down_bias_text = f"{rollout_metrics['down_bias']:.2f}"
                    non_right_direction_rate_text = f"{rollout_metrics['non_right_direction_rate']:.2f}"
                    dominant_action_share_text = f"{rollout_metrics['dominant_action_share']:.2f}"
                    direction_entropy_norm_text = f"{rollout_metrics['direction_entropy_norm']:.2f}"
                    vertical_bias_text = f"{rollout_metrics['vertical_bias']:.2f}"
                    invalid_ball_skill_rate_text = f"{rollout_metrics['invalid_ball_skill_rate']:.2f}"
                    invalid_no_ball_pass_rate_text = f"{rollout_metrics['invalid_no_ball_pass_rate']:.2f}"
                    invalid_no_ball_shot_rate_text = f"{rollout_metrics['invalid_no_ball_shot_rate']:.2f}"
                    print(
                        f"[update {update}/{num_updates}] "
                        f"envs={self.env.num_envs} "
                        f"agent_steps={self.total_agent_steps} "
                        f"env_steps={self.total_env_steps} "
                        f"model={self.config.model_type} "
                        f"policy_loss={update_metrics['policy_loss']:.4f} "
                        f"value_loss={update_metrics['value_loss']:.4f} "
                        f"entropy={update_metrics['entropy']:.4f} "
                        f"approx_kl={update_metrics['approx_kl']:.5f} "
                        f"episodes_finished={int(rollout_metrics['episodes_finished'])} "
                        f"episode_return={return_text} "
                        f"score_reward={score_reward_text} "
                        f"pass_attempts_ep={mean_pass_attempts_text} "
                        f"pass_successes_ep={mean_pass_successes_text} "
                        f"pass_success_per_attempt={pass_success_per_attempt_text} "
                        f"pass_progress_ep={mean_pass_progress_text} "
                        f"final_third_entries_ep={mean_final_third_entries_text} "
                        f"shot_attempt_events_ep={mean_shot_attempt_events_text} "
                        f"shot_per_final_third_entry={shot_per_final_third_entry_text} "
                        f"own_half_turnovers_ep={mean_own_half_turnovers_text} "
                        f"possession_recoveries_ep={mean_possession_recoveries_text} "
                        f"defensive_third_recoveries_ep={mean_defensive_third_recoveries_text} "
                        f"opponent_dangerous_possessions_ep={mean_opponent_dangerous_possessions_text} "
                        f"goals_for={goals_for_text} "
                        f"goals_against={goals_against_text} "
                        f"score_examples={rollout_metrics['score_examples']} "
                        f"pass_rate={pass_rate_text} "
                        f"long_pass_rate={long_pass_rate_text} "
                        f"high_pass_rate={high_pass_rate_text} "
                        f"short_pass_rate={short_pass_rate_text} "
                        f"shot_rate={shot_rate_text} "
                        f"skill_rate={skill_rate_text} "
                        f"idle_rate={idle_rate_text} "
                        f"release_direction_rate={release_direction_rate_text} "
                        f"sprint_rate={sprint_rate_text} "
                        f"dribble_rate={dribble_rate_text} "
                        f"direction_rate={direction_rate_text} "
                        f"right_bias={right_bias_text} "
                        f"left_bias={left_bias_text} "
                        f"up_bias={up_bias_text} "
                        f"down_bias={down_bias_text} "
                        f"non_right_direction_rate={non_right_direction_rate_text} "
                        f"dominant_action_share={dominant_action_share_text} "
                        f"direction_entropy_norm={direction_entropy_norm_text} "
                        f"vertical_bias={vertical_bias_text} "
                        f"invalid_ball_skill_rate={invalid_ball_skill_rate_text} "
                        f"invalid_no_ball_pass_rate={invalid_no_ball_pass_rate_text} "
                        f"invalid_no_ball_shot_rate={invalid_no_ball_shot_rate_text} "
                        f"top_actions={rollout_metrics['top_actions']} "
                        f"episode_length={length_text} "
                        f"episode_length_range={length_range_text} "
                        f"success_rate={success_text} "
                        f"env_fps={env_steps_per_update / rollout_time:.1f} "
                        f"samples_per_sec={agent_steps_per_update / rollout_time:.1f} "
                        f"update_sec={update_time:.2f} "
                        f"total_sps={self.total_agent_steps / total_elapsed:.1f}"
                    )

                if update % self.config.save_interval == 0:
                    latest_checkpoint = self.save_checkpoint(f"update_{update}.pt", update)
        finally:
            self.env.close()

        latest_checkpoint = self.save_checkpoint("latest.pt", num_updates)
        print(f"Training finished. Latest checkpoint: {latest_checkpoint}")
        return latest_checkpoint


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a shared-policy PPO agent on Google Research Football.")
    parser.add_argument("--preset", choices=sorted(PRESET_OVERRIDES.keys()), default="default")
    parser.add_argument("--env-name", default="11_vs_11_easy_stochastic")
    parser.add_argument("--representation", default="simple115v2")
    parser.add_argument("--rewards", default="scoring,checkpoints")
    parser.add_argument("--num-controlled-players", type=int, default=11)
    parser.add_argument("--channel-width", type=int, default=42)
    parser.add_argument("--channel-height", type=int, default=42)
    parser.add_argument("--model-type", choices=MODEL_TYPES, default="auto")
    parser.add_argument("--feature-dim", type=int, default=256)
    parser.add_argument("--use-player-id", action="store_true")
    parser.add_argument("--total-timesteps", type=int, default=500_000)
    parser.add_argument("--rollout-steps", type=int, default=256)
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--update-epochs", type=int, default=4)
    parser.add_argument("--num-minibatches", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-coef", type=float, default=0.2)
    parser.add_argument("--ent-coef", type=float, default=0.01)
    parser.add_argument("--vf-coef", type=float, default=0.5)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--hidden-sizes", type=int, nargs="+", default=[256, 256])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--save-interval", type=int, default=10)
    parser.add_argument("--log-interval", type=int, default=1)
    parser.add_argument("--save-dir", default="Y_Fu/checkpoints")
    parser.add_argument("--logdir", default="Y_Fu/logs")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--init-checkpoint")
    parser.add_argument("--pass-success-reward", type=float, default=0.0)
    parser.add_argument("--pass-failure-penalty", type=float, default=0.0)
    parser.add_argument("--pass-progress-reward-scale", type=float, default=0.0)
    parser.add_argument("--shot-attempt-reward", type=float, default=0.0)
    parser.add_argument("--attacking-possession-reward", type=float, default=0.0)
    parser.add_argument("--attacking-x-threshold", type=float, default=0.55)
    parser.add_argument("--final-third-entry-reward", type=float, default=0.0)
    parser.add_argument("--possession-retention-reward", type=float, default=0.0)
    parser.add_argument("--possession-recovery-reward", type=float, default=0.0)
    parser.add_argument("--defensive-third-recovery-reward", type=float, default=0.0)
    parser.add_argument("--opponent-attacking-possession-penalty", type=float, default=0.0)
    parser.add_argument("--own-half-turnover-penalty", type=float, default=0.0)
    parser.add_argument("--own-half-x-threshold", type=float, default=0.0)
    parser.add_argument("--defensive-x-threshold", type=float, default=-0.45)
    parser.add_argument("--pending-pass-horizon", type=int, default=8)
    return parser


def _build_default_arg_values() -> dict[str, Any]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--preset", choices=sorted(PRESET_OVERRIDES.keys()), default="default")
    parser.add_argument("--env-name", default="11_vs_11_easy_stochastic")
    parser.add_argument("--representation", default="simple115v2")
    parser.add_argument("--rewards", default="scoring,checkpoints")
    parser.add_argument("--num-controlled-players", type=int, default=11)
    parser.add_argument("--channel-width", type=int, default=42)
    parser.add_argument("--channel-height", type=int, default=42)
    parser.add_argument("--model-type", choices=MODEL_TYPES, default="auto")
    parser.add_argument("--feature-dim", type=int, default=256)
    parser.add_argument("--use-player-id", action="store_true")
    parser.add_argument("--total-timesteps", type=int, default=500_000)
    parser.add_argument("--rollout-steps", type=int, default=256)
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--update-epochs", type=int, default=4)
    parser.add_argument("--num-minibatches", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-coef", type=float, default=0.2)
    parser.add_argument("--ent-coef", type=float, default=0.01)
    parser.add_argument("--vf-coef", type=float, default=0.5)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--hidden-sizes", type=int, nargs="+", default=[256, 256])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--save-interval", type=int, default=10)
    parser.add_argument("--log-interval", type=int, default=1)
    parser.add_argument("--save-dir", default="Y_Fu/checkpoints")
    parser.add_argument("--logdir", default="Y_Fu/logs")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--init-checkpoint")
    parser.add_argument("--pass-success-reward", type=float, default=0.0)
    parser.add_argument("--pass-failure-penalty", type=float, default=0.0)
    parser.add_argument("--pass-progress-reward-scale", type=float, default=0.0)
    parser.add_argument("--shot-attempt-reward", type=float, default=0.0)
    parser.add_argument("--attacking-possession-reward", type=float, default=0.0)
    parser.add_argument("--attacking-x-threshold", type=float, default=0.55)
    parser.add_argument("--final-third-entry-reward", type=float, default=0.0)
    parser.add_argument("--possession-retention-reward", type=float, default=0.0)
    parser.add_argument("--possession-recovery-reward", type=float, default=0.0)
    parser.add_argument("--defensive-third-recovery-reward", type=float, default=0.0)
    parser.add_argument("--opponent-attacking-possession-penalty", type=float, default=0.0)
    parser.add_argument("--own-half-turnover-penalty", type=float, default=0.0)
    parser.add_argument("--own-half-x-threshold", type=float, default=0.0)
    parser.add_argument("--defensive-x-threshold", type=float, default=-0.45)
    parser.add_argument("--pending-pass-horizon", type=int, default=8)
    return vars(parser.parse_args([]))


def config_from_args(args: argparse.Namespace) -> PPOConfig:
    defaults = _build_default_arg_values()
    merged: dict[str, Any] = dict(PRESET_OVERRIDES[args.preset])

    for key, value in vars(args).items():
        if key == "preset":
            continue
        if key not in merged or value != defaults[key]:
            merged[key] = value

    merged["channel_dimensions"] = (merged.pop("channel_width"), merged.pop("channel_height"))
    merged["hidden_sizes"] = tuple(merged["hidden_sizes"])
    merged["num_controlled_players"] = merged["num_controlled_players"]
    return PPOConfig(**merged)


def main(argv: Sequence[str] | None = None) -> Path:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    trainer = PPOTrainer(config_from_args(args))
    return trainer.train()


if __name__ == "__main__":
    main()
