from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

GOALKEEPER_ROLE = 0


@dataclass(frozen=True)
class RewardShapingConfig:
    enabled: bool = False
    defensive_ball_chasing_reward_scale: float = 0.0
    goalkeeper_overextension_penalty: float = 0.0
    goalkeeper_max_x: float = -0.35
    defensive_recovery_reward: float = 0.0
    defensive_x_threshold: float = 0.15


def _primary_observation(raw_observation: Any) -> dict[str, Any] | None:
    if isinstance(raw_observation, list) and raw_observation:
        first = raw_observation[0]
        if isinstance(first, dict):
            return first
    if isinstance(raw_observation, dict):
        return raw_observation
    return None


def _left_team_xy(raw_observation: dict[str, Any]) -> np.ndarray:
    team = np.asarray(raw_observation.get("left_team", []), dtype=np.float32)
    if team.ndim != 2 or team.shape[0] == 0:
        return np.zeros((0, 2), dtype=np.float32)
    return team[:, :2]


def _left_team_roles(raw_observation: dict[str, Any], expected_size: int) -> np.ndarray:
    roles = np.asarray(raw_observation.get("left_team_roles", np.zeros(expected_size)), dtype=np.int64)
    if roles.ndim != 1 or roles.size != expected_size:
        return np.zeros(expected_size, dtype=np.int64)
    return roles


def _safe_player_index(raw_observation: dict[str, Any], team: np.ndarray) -> int:
    if team.shape[0] == 0:
        return 0
    try:
        idx = int(raw_observation.get("active", 0))
    except (TypeError, ValueError):
        idx = 0
    return int(np.clip(idx, 0, team.shape[0] - 1))


def _goalkeeper_index(raw_observation: dict[str, Any], team: np.ndarray) -> int:
    if team.shape[0] == 0:
        return 0
    roles = _left_team_roles(raw_observation, team.shape[0])
    goalkeeper_candidates = np.flatnonzero(roles == GOALKEEPER_ROLE)
    if goalkeeper_candidates.size > 0:
        return int(goalkeeper_candidates[0])
    return 0


def _ball_xy(raw_observation: dict[str, Any]) -> np.ndarray:
    ball = np.asarray(raw_observation.get("ball", [0.0, 0.0, 0.0]), dtype=np.float32)
    if ball.size < 2:
        return np.zeros(2, dtype=np.float32)
    return ball[:2]


def extract_step_metrics(raw_observation: Any) -> dict[str, float]:
    default_metrics = {
        "goalkeeper_x": float("nan"),
        "active_player_distance_to_ball": float("nan"),
    }

    obs = _primary_observation(raw_observation)
    if obs is None:
        return default_metrics

    left_team = _left_team_xy(obs)
    if left_team.shape[0] == 0:
        return default_metrics

    active_index = _safe_player_index(obs, left_team)
    goalkeeper_index = _goalkeeper_index(obs, left_team)
    ball_xy = _ball_xy(obs)

    return {
        "goalkeeper_x": float(left_team[goalkeeper_index, 0]),
        "active_player_distance_to_ball": float(np.linalg.norm(left_team[active_index] - ball_xy)),
    }


def compute_shaped_reward(
    previous_raw_observation: Any,
    next_raw_observation: Any,
    config: RewardShapingConfig,
) -> tuple[float, dict[str, float]]:
    if not config.enabled:
        return 0.0, {}

    previous = _primary_observation(previous_raw_observation)
    current = _primary_observation(next_raw_observation)
    if previous is None or current is None:
        return 0.0, {}

    prev_team = _left_team_xy(previous)
    current_team = _left_team_xy(current)
    if prev_team.shape[0] == 0 or current_team.shape[0] == 0:
        return 0.0, {}

    prev_ball_xy = _ball_xy(previous)
    current_ball_xy = _ball_xy(current)
    prev_ball_x = float(prev_ball_xy[0])
    current_ball_x = float(current_ball_xy[0])
    prev_owned_team = int(previous.get("ball_owned_team", -1))
    current_owned_team = int(current.get("ball_owned_team", -1))

    reward_bonus = 0.0
    shaping_info: dict[str, float] = {}

    if (
        config.defensive_ball_chasing_reward_scale != 0.0
        and prev_owned_team != 0
        and prev_ball_x <= max(0.15, config.defensive_x_threshold + 0.15)
    ):
        prev_active_index = _safe_player_index(previous, prev_team)
        current_active_index = _safe_player_index(current, current_team)
        prev_distance = float(np.linalg.norm(prev_team[prev_active_index] - prev_ball_xy))
        current_distance = float(np.linalg.norm(current_team[current_active_index] - current_ball_xy))
        closing_gain = max(0.0, prev_distance - current_distance)
        if closing_gain > 0.0:
            chase_bonus = closing_gain * config.defensive_ball_chasing_reward_scale
            reward_bonus += chase_bonus
            shaping_info["ball_chasing_bonus"] = chase_bonus

    if config.goalkeeper_overextension_penalty != 0.0:
        goalkeeper_index = _goalkeeper_index(current, current_team)
        goalkeeper_x = float(current_team[goalkeeper_index, 0])
        excess_x = max(0.0, goalkeeper_x - config.goalkeeper_max_x)
        if excess_x > 0.0:
            penalty = excess_x * config.goalkeeper_overextension_penalty
            if current_ball_x > 0.0:
                penalty *= 1.5
            elif current_ball_x > config.defensive_x_threshold:
                penalty *= 1.2
            reward_bonus -= penalty
            shaping_info["goalkeeper_penalty"] = -penalty

    if (
        config.defensive_recovery_reward != 0.0
        and prev_owned_team != 0
        and current_owned_team == 0
        and prev_ball_x <= config.defensive_x_threshold
    ):
        reward_bonus += config.defensive_recovery_reward
        shaping_info["defensive_recovery_bonus"] = config.defensive_recovery_reward

    return reward_bonus, shaping_info
