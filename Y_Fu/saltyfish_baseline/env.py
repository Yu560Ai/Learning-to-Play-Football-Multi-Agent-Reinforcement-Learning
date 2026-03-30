from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .features import ENGINEERED_FEATURE_DIM, build_engineered_features
from yfu_football.envs import FootballEnvWrapper

# Default action_set_v1 indices with sliding removed.
REDUCED_ACTION_INDICES: tuple[int, ...] = (
    0,  # idle
    1, 2, 3, 4, 5, 6, 7, 8,  # directions
    9, 10, 11,  # passes
    12,  # shot
    13, 14, 15,  # sprint + releases
    17, 18,  # dribble + release
)


@dataclass(frozen=True)
class CompetitionEnvConfig:
    env_name: str = "11_vs_11_kaggle"
    representation: str = "simple115v2"
    rewards: str = "scoring"
    num_controlled_players: int = 1
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
    """Single-player competition-style wrapper with reduced action set."""

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
        if isinstance(action, (np.ndarray, list, tuple)):
            action_index = int(np.asarray(action).reshape(-1)[0])
        else:
            action_index = int(action)
        mapped_action = int(self.action_map[action_index])
        observation, reward, done, info = self.base_env.step(mapped_action)
        next_raw = self.get_raw_observation()
        augmented_observation = self._augment_observation(observation, raw_obs=next_raw)
        shaped_reward, shaping_metrics = self._compute_shaped_reward(previous_raw, next_raw, mapped_action)
        if shaped_reward != 0.0:
            reward = np.asarray(reward, dtype=np.float32) + shaped_reward
        info = dict(info)
        info.update(shaping_metrics)
        return augmented_observation, reward, done, info

    def close(self) -> None:
        self.base_env.close()

    def sample_random_action(self) -> int:
        return int(np.random.randint(self.action_dim))

    def get_score(self) -> tuple[int, int] | None:
        return self.base_env.get_score()

    def get_raw_observation(self) -> Any:
        return self.base_env.get_raw_observation()

    def _augment_observation(self, observation: np.ndarray, *, raw_obs: Any | None = None) -> np.ndarray:
        array = np.asarray(observation, dtype=np.float32).reshape(1, self.base_obs_dim)
        if not self.config.use_engineered_features:
            return array
        raw = raw_obs if raw_obs is not None else self.get_raw_observation()
        raw_single = raw[0] if isinstance(raw, list) else raw
        engineered = build_engineered_features(raw_single).reshape(1, ENGINEERED_FEATURE_DIM)
        return np.concatenate([array, engineered], axis=1)

    def _compute_shaped_reward(self, previous_raw: Any, next_raw: Any, mapped_action: int) -> tuple[float, dict[str, float]]:
        previous = previous_raw[0] if isinstance(previous_raw, list) else previous_raw
        current = next_raw[0] if isinstance(next_raw, list) else next_raw
        if previous is None or current is None:
            return 0.0, {}

        reward_cfg = self.config.reward_config
        prev_team = int(previous.get("ball_owned_team", -1))
        next_team = int(current.get("ball_owned_team", -1))
        prev_player = int(previous.get("ball_owned_player", -1))
        next_player = int(current.get("ball_owned_player", -1))
        prev_game_mode = int(previous.get("game_mode", 0))
        next_game_mode = int(current.get("game_mode", 0))
        prev_ball_x = float(np.asarray(previous["ball"], dtype=np.float32)[0])
        next_ball_x = float(np.asarray(current["ball"], dtype=np.float32)[0])
        attacking_third = 1.0 if next_team == 0 and next_ball_x >= 0.5 else 0.0
        shot_action = 1.0 if mapped_action == 12 else 0.0
        shot_with_ball = 1.0 if mapped_action == 12 and prev_team == 0 else 0.0
        pass_completed = 1.0 if (
            mapped_action in (9, 10, 11) and prev_team == 0 and next_team == 0 and next_player >= 0 and next_player != prev_player
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
        reward = possession_reward + pass_reward + carry_reward + territory_reward + shot_reward - out_penalty

        metrics = {
            "possession_gains": 1.0 if prev_team != 0 and next_team == 0 else 0.0,
            "possession_losses": 1.0 if prev_team == 0 and next_team != 0 else 0.0,
            "team_possession_steps": 1.0 if next_team == 0 else 0.0,
            "opponent_possession_steps": 1.0 if next_team == 1 else 0.0,
            "successful_passes": pass_completed,
            "forward_ball_progress": forward_ball_progress,
            "carry_progress": carry_progress,
            "attacking_third_possession_steps": attacking_third,
            "shot_actions": shot_action,
            "shots_with_ball": shot_with_ball,
            "ball_out_losses": out_of_play_loss,
            "possession_reward": float(possession_reward),
            "pass_reward": float(pass_reward),
            "carry_reward": float(carry_reward),
            "territory_reward": float(territory_reward),
            "shot_reward": float(shot_reward),
            "out_penalty": float(out_penalty),
            "shaping_reward": float(reward),
        }
        return float(reward), metrics
