from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .features import ENGINEERED_FEATURE_DIM, build_engineered_features


def _ensure_yfu_on_path() -> None:
    root = Path(__file__).resolve().parents[2]
    yfu_dir = root / "Y_Fu"
    yfu_dir_str = str(yfu_dir)
    if yfu_dir_str not in sys.path:
        sys.path.insert(0, yfu_dir_str)


_ensure_yfu_on_path()
from yfu_football.envs import FootballEnvWrapper  # type: ignore  # noqa: E402


REDUCED_ACTION_INDICES: tuple[int, ...] = (
    0,
    1, 2, 3, 4, 5, 6, 7, 8,
    9, 10, 11,
    12,
    13, 14, 15,
    17, 18,
)


@dataclass(frozen=True)
class CompetitionEnvConfig:
    env_name: str = "5_vs_5"
    representation: str = "simple115v2"
    rewards: str = "scoring"
    num_controlled_players: int = 4
    channel_dimensions: tuple[int, int] = (42, 42)
    use_engineered_features: bool = True
    reward_config: "CompetitionRewardConfig" = field(default_factory=lambda: CompetitionRewardConfig())


@dataclass(frozen=True)
class CompetitionRewardConfig:
    possession_gain_reward: float = 0.2
    possession_loss_penalty: float = 0.2
    team_possession_reward: float = 0.001
    opponent_possession_penalty: float = 0.001
    successful_pass_reward: float = 0.02
    progressive_pass_reward_scale: float = 0.05
    carry_progress_reward_scale: float = 0.04
    attacking_third_reward: float = 0.002
    shots_with_ball_reward: float = 0.01
    attacking_risk_x_threshold: float = 0.4
    attacking_loss_penalty_scale: float = 0.5
    out_of_play_loss_penalty: float = 0.05


class ReducedActionFootballEnv:
    def __init__(
        self,
        config: CompetitionEnvConfig | None = None,
        *,
        render: bool = False,
        write_video: bool = False,
        logdir: str | None = None,
        other_config_options: dict[str, Any] | None = None,
    ) -> None:
        self.config = config or CompetitionEnvConfig()
        self.base_env = FootballEnvWrapper(
            env_name=self.config.env_name,
            representation=self.config.representation,
            rewards=self.config.rewards,
            render=render,
            write_video=write_video,
            logdir=logdir,
            num_controlled_players=self.config.num_controlled_players,
            channel_dimensions=self.config.channel_dimensions,
            other_config_options=other_config_options or {},
        )
        self.action_map = np.asarray(REDUCED_ACTION_INDICES, dtype=np.int64)
        self.base_obs_dim = self.base_env.obs_dim
        self.base_obs_shape = self.base_env.obs_shape
        self.obs_dim = self.base_obs_dim + (ENGINEERED_FEATURE_DIM if self.config.use_engineered_features else 0)
        self.obs_shape = (self.obs_dim,)
        self.action_dim = int(self.action_map.shape[0])
        self.num_players = self.base_env.num_players

    def reset(self) -> np.ndarray:
        observation = self.base_env.reset()
        return self._augment_observation(observation)

    def step(self, action: int | np.ndarray | list[int]) -> tuple[np.ndarray, np.ndarray, bool, dict[str, Any]]:
        previous_raw = self.get_raw_observation()
        action_indices = self._normalize_actions(action)
        mapped_action = self.action_map[action_indices]
        observation, reward, done, info = self.base_env.step(mapped_action)
        next_raw = self.get_raw_observation()
        done = bool(done or self._is_terminal_from_raw(next_raw))
        augmented_observation = self._augment_observation(observation, raw_obs=next_raw)
        shaped_reward, shaping_metrics = self._compute_shaped_reward(previous_raw, next_raw, mapped_action)
        reward_array = np.asarray(reward, dtype=np.float32).reshape(self.num_players)
        reward_array = reward_array + shaped_reward
        info = dict(info or {})
        info.update(shaping_metrics)
        info["steps_left"] = self._mean_steps_left(next_raw)
        return augmented_observation, reward_array, done, info

    def close(self) -> None:
        self.base_env.close()

    def sample_random_action(self) -> np.ndarray:
        return np.random.randint(self.action_dim, size=self.num_players, dtype=np.int64)

    def get_score(self) -> tuple[int, int] | None:
        return self.base_env.get_score()

    def get_raw_observation(self) -> Any:
        return self.base_env.get_raw_observation()

    def _normalize_actions(self, action: int | np.ndarray | list[int]) -> np.ndarray:
        if isinstance(action, np.ndarray):
            action_array = action.astype(np.int64, copy=False).reshape(-1)
        elif isinstance(action, list):
            action_array = np.asarray(action, dtype=np.int64).reshape(-1)
        else:
            action_array = np.asarray([int(action)], dtype=np.int64)
        if action_array.size == 1 and self.num_players > 1:
            action_array = np.repeat(action_array, self.num_players)
        return action_array[: self.num_players]

    def _augment_observation(self, observation: np.ndarray, *, raw_obs: Any | None = None) -> np.ndarray:
        array = np.asarray(observation, dtype=np.float32).reshape(self.num_players, self.base_obs_dim)
        if not self.config.use_engineered_features:
            return array
        raw = raw_obs if raw_obs is not None else self.get_raw_observation()
        raw_list = raw if isinstance(raw, list) else [raw]
        features: list[np.ndarray] = []
        for idx in range(self.num_players):
            raw_single = raw_list[idx] if idx < len(raw_list) else raw_list[0]
            engineered = build_engineered_features(raw_single)
            features.append(engineered.astype(np.float32))
        return np.concatenate([array, np.stack(features, axis=0)], axis=1)

    def _compute_shaped_reward(self, previous_raw: Any, next_raw: Any, mapped_action: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
        prev_list = previous_raw if isinstance(previous_raw, list) else [previous_raw]
        next_list = next_raw if isinstance(next_raw, list) else [next_raw]
        reward_cfg = self.config.reward_config
        rewards = np.zeros(self.num_players, dtype=np.float32)
        metrics_total = {
            "possession_gains": 0.0,
            "possession_losses": 0.0,
            "team_possession_steps": 0.0,
            "opponent_possession_steps": 0.0,
            "successful_passes": 0.0,
            "forward_ball_progress": 0.0,
            "carry_progress": 0.0,
            "attacking_third_possession_steps": 0.0,
            "shot_actions": 0.0,
            "shots_with_ball": 0.0,
            "ball_out_losses": 0.0,
            "possession_reward": 0.0,
            "pass_reward": 0.0,
            "carry_reward": 0.0,
            "territory_reward": 0.0,
            "shot_reward": 0.0,
            "out_penalty": 0.0,
            "shaping_reward": 0.0,
        }

        for idx in range(self.num_players):
            previous = prev_list[idx] if idx < len(prev_list) else prev_list[0]
            current = next_list[idx] if idx < len(next_list) else next_list[0]
            if previous is None or current is None:
                continue

            prev_team = int(previous.get("ball_owned_team", -1))
            next_team = int(current.get("ball_owned_team", -1))
            prev_player = int(previous.get("ball_owned_player", -1))
            next_player = int(current.get("ball_owned_player", -1))
            next_game_mode = int(current.get("game_mode", 0))
            prev_ball_x = float(np.asarray(previous["ball"], dtype=np.float32)[0])
            next_ball_x = float(np.asarray(current["ball"], dtype=np.float32)[0])
            attacking_third = 1.0 if next_team == 0 and next_ball_x >= 0.5 else 0.0
            shot_action = 1.0 if int(mapped_action[idx]) == 12 else 0.0
            shot_with_ball = 1.0 if int(mapped_action[idx]) == 12 and prev_team == 0 else 0.0
            pass_completed = 1.0 if (
                int(mapped_action[idx]) in (9, 10, 11)
                and prev_team == 0
                and next_team == 0
                and next_player >= 0
                and next_player != prev_player
            ) else 0.0
            forward_ball_progress = max(0.0, next_ball_x - prev_ball_x)
            same_player_carry = 1.0 if prev_team == 0 and next_team == 0 and prev_player >= 0 and prev_player == next_player else 0.0
            carry_progress = forward_ball_progress * same_player_carry
            attacking_loss_scale = (
                reward_cfg.attacking_loss_penalty_scale
                if prev_team == 0 and prev_ball_x >= reward_cfg.attacking_risk_x_threshold
                else 1.0
            )
            out_of_play_loss = 1.0 if prev_team == 0 and next_team == -1 and next_game_mode in (2, 5) else 0.0

            possession_reward = 0.0
            if prev_team != 0 and next_team == 0:
                possession_reward += reward_cfg.possession_gain_reward
            if prev_team == 0 and next_team != 0:
                possession_reward -= reward_cfg.possession_loss_penalty * attacking_loss_scale
            if next_team == 0:
                possession_reward += reward_cfg.team_possession_reward
            elif next_team == 1:
                possession_reward -= reward_cfg.opponent_possession_penalty

            pass_reward = 0.0
            if pass_completed:
                pass_reward += reward_cfg.successful_pass_reward
                pass_reward += forward_ball_progress * reward_cfg.progressive_pass_reward_scale

            carry_reward = carry_progress * reward_cfg.carry_progress_reward_scale
            territory_reward = attacking_third * reward_cfg.attacking_third_reward
            shot_reward = shot_with_ball * reward_cfg.shots_with_ball_reward
            out_penalty = out_of_play_loss * reward_cfg.out_of_play_loss_penalty
            shaped = possession_reward + pass_reward + carry_reward + territory_reward + shot_reward - out_penalty
            rewards[idx] = float(shaped)

            metrics_total["possession_gains"] += 1.0 if prev_team != 0 and next_team == 0 else 0.0
            metrics_total["possession_losses"] += 1.0 if prev_team == 0 and next_team != 0 else 0.0
            metrics_total["team_possession_steps"] += 1.0 if next_team == 0 else 0.0
            metrics_total["opponent_possession_steps"] += 1.0 if next_team == 1 else 0.0
            metrics_total["successful_passes"] += pass_completed
            metrics_total["forward_ball_progress"] += forward_ball_progress
            metrics_total["carry_progress"] += carry_progress
            metrics_total["attacking_third_possession_steps"] += attacking_third
            metrics_total["shot_actions"] += shot_action
            metrics_total["shots_with_ball"] += shot_with_ball
            metrics_total["ball_out_losses"] += out_of_play_loss
            metrics_total["possession_reward"] += float(possession_reward)
            metrics_total["pass_reward"] += float(pass_reward)
            metrics_total["carry_reward"] += float(carry_reward)
            metrics_total["territory_reward"] += float(territory_reward)
            metrics_total["shot_reward"] += float(shot_reward)
            metrics_total["out_penalty"] += float(out_penalty)
            metrics_total["shaping_reward"] += float(shaped)

        return rewards, metrics_total

    def _mean_steps_left(self, raw_observation: Any) -> float:
        raw_list = raw_observation if isinstance(raw_observation, list) else [raw_observation]
        values: list[float] = []
        for item in raw_list:
            if isinstance(item, dict) and "steps_left" in item:
                try:
                    values.append(float(item["steps_left"]))
                except (TypeError, ValueError):
                    continue
        return float(np.mean(values)) if values else float("nan")

    def _is_terminal_from_raw(self, raw_observation: Any) -> bool:
        raw_list = raw_observation if isinstance(raw_observation, list) else [raw_observation]
        for item in raw_list:
            if not isinstance(item, dict):
                continue
            try:
                if float(item.get("steps_left", 1.0)) <= 0.0:
                    return True
            except (TypeError, ValueError):
                continue
        return False
