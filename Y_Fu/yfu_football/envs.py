from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


PASS_ACTIONS = {9, 10, 11}
SHOT_ACTION = 12


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _football_source_root() -> Path:
    return _project_root() / "football-master"


def ensure_local_gfootball_path() -> Path:
    football_root = _football_source_root()
    football_root_str = str(football_root)
    if football_root.exists() and football_root_str not in sys.path:
        sys.path.insert(0, football_root_str)
    return football_root


def import_football_env():
    ensure_local_gfootball_path()
    try:
        import gfootball.env as football_env  # type: ignore
    except ImportError as exc:
        football_root = _football_source_root()
        extra_hint = ""
        if "gfootball_engine" in str(exc):
            extra_hint = (
                " The Python package path is visible, but the compiled "
                "'gfootball_engine' module is missing, so the football engine "
                "still needs to be built."
            )
        raise ImportError(
            "Could not import gfootball. Expected the local source tree at "
            f"'{football_root}'. If the package still cannot be imported, build "
            f"the environment first inside football-master.{extra_hint}"
        ) from exc
    return football_env


@dataclass
class EnvSpec:
    obs_dim: int
    obs_shape: tuple[int, ...]
    action_dim: int
    num_players: int


@dataclass(frozen=True)
class RewardShapingConfig:
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


class FootballEnvWrapper:
    def __init__(
        self,
        env_name: str = "11_vs_11_easy_stochastic",
        representation: str = "simple115v2",
        rewards: str = "scoring,checkpoints",
        stacked: bool = False,
        render: bool = False,
        write_video: bool = False,
        logdir: str | None = None,
        num_controlled_players: int = 11,
        channel_dimensions: tuple[int, int] = (42, 42),
        other_config_options: dict[str, Any] | None = None,
        reward_shaping: RewardShapingConfig | None = None,
    ) -> None:
        self._requested_num_players = num_controlled_players
        self.reward_shaping = reward_shaping or RewardShapingConfig()
        football_env = import_football_env()
        self.env = football_env.create_environment(
            env_name=env_name,
            stacked=stacked,
            representation=representation,
            rewards=rewards,
            render=render,
            write_video=write_video,
            logdir=logdir or "",
            number_of_left_players_agent_controls=num_controlled_players,
            channel_dimensions=channel_dimensions,
            other_config_options=other_config_options or {},
        )
        self.spec = self._infer_spec()
        self._pending_pass: dict[str, Any] | None = None

    def _infer_spec(self) -> EnvSpec:
        observation_space = self.env.observation_space
        action_space = self.env.action_space

        full_obs_shape = tuple(int(dim) for dim in observation_space.shape)
        if self._requested_num_players == 1:
            num_players = 1
            obs_shape = full_obs_shape
            obs_dim = int(np.prod(obs_shape))
        elif len(full_obs_shape) == 1:
            num_players = 1
            obs_shape = full_obs_shape
            obs_dim = obs_shape[0]
        else:
            num_players = full_obs_shape[0]
            obs_shape = full_obs_shape[1:]
            obs_dim = int(np.prod(obs_shape))

        if hasattr(action_space, "n"):
            action_dim = int(action_space.n)
        elif hasattr(action_space, "nvec"):
            action_dim = int(action_space.nvec[0])
        else:
            raise TypeError(f"Unsupported action space: {action_space!r}")

        return EnvSpec(obs_dim=obs_dim, obs_shape=obs_shape, action_dim=action_dim, num_players=num_players)

    @property
    def obs_dim(self) -> int:
        return self.spec.obs_dim

    @property
    def action_dim(self) -> int:
        return self.spec.action_dim

    @property
    def num_players(self) -> int:
        return self.spec.num_players

    @property
    def obs_shape(self) -> tuple[int, ...]:
        return self.spec.obs_shape

    def reset(self) -> np.ndarray:
        self._pending_pass = None
        observation = self.env.reset()
        return self._format_observation(observation)

    def step(self, actions: np.ndarray | list[int] | int) -> tuple[np.ndarray, np.ndarray, bool, dict[str, Any]]:
        action_input = self._format_action(actions)
        previous_raw = self.get_raw_observation()
        observation, reward, done, info = self.env.step(action_input)
        next_raw = self.get_raw_observation()
        formatted_obs = self._format_observation(observation)
        formatted_reward = self._format_reward(reward)
        shaped_reward, shaping_info = self._compute_custom_reward(
            previous_raw=previous_raw,
            next_raw=next_raw,
            actions=action_input,
            done=bool(done),
        )
        if shaped_reward != 0.0:
            formatted_reward = formatted_reward + shaped_reward
        if shaping_info:
            info = dict(info)
            info.update(shaping_info)
        return formatted_obs, formatted_reward, bool(done), info

    def close(self) -> None:
        if hasattr(self.env, "close"):
            self.env.close()

    def sample_random_action(self) -> np.ndarray | int:
        sample = self.env.action_space.sample()
        if self.num_players == 1:
            return int(sample)
        return np.asarray(sample, dtype=np.int64).reshape(self.num_players)

    def get_raw_observation(self) -> Any:
        base_env = getattr(self.env, "unwrapped", self.env)
        if hasattr(base_env, "observation"):
            return base_env.observation()
        if hasattr(self.env, "observation"):
            return self.env.observation()
        return None

    def get_score(self) -> tuple[int, int] | None:
        raw_observation = self.get_raw_observation()
        if isinstance(raw_observation, list) and raw_observation and "score" in raw_observation[0]:
            left_score, right_score = raw_observation[0]["score"]
            return int(left_score), int(right_score)
        return None

    def _format_observation(self, observation: Any) -> np.ndarray:
        array = np.asarray(observation, dtype=np.float32)
        if self.num_players == 1:
            return array.reshape(1, self.obs_dim)
        if array.ndim == 1:
            array = np.expand_dims(array, axis=0)
        return array.reshape(self.num_players, self.obs_dim)

    def _format_reward(self, reward: Any) -> np.ndarray:
        array = np.asarray(reward, dtype=np.float32)
        if array.ndim == 0:
            array = np.expand_dims(array, axis=0)
        return array.reshape(self.num_players)

    def _format_action(self, actions: np.ndarray | list[int] | int) -> list[int] | int:
        if self.num_players == 1:
            if isinstance(actions, (list, tuple, np.ndarray)):
                return int(np.asarray(actions).reshape(-1)[0])
            return int(actions)
        return np.asarray(actions, dtype=np.int64).reshape(self.num_players).tolist()

    def _primary_raw_observation(self, raw_observation: Any) -> dict[str, Any] | None:
        if isinstance(raw_observation, list) and raw_observation:
            first = raw_observation[0]
            if isinstance(first, dict):
                return first
        return None

    def _ball_x(self, raw_observation: dict[str, Any]) -> float:
        ball = raw_observation.get("ball", [0.0])
        if isinstance(ball, (list, tuple, np.ndarray)) and len(ball) > 0:
            return float(ball[0])
        return 0.0

    def _compute_custom_reward(
        self,
        previous_raw: Any,
        next_raw: Any,
        actions: list[int] | int,
        done: bool,
    ) -> tuple[float, dict[str, float]]:
        if self.reward_shaping == RewardShapingConfig():
            return 0.0, {}

        prev = self._primary_raw_observation(previous_raw)
        nxt = self._primary_raw_observation(next_raw)
        if prev is None or nxt is None:
            self._pending_pass = None
            return 0.0, {}

        reward_bonus = 0.0
        shaping_info: dict[str, float] = {}
        action_list = [int(actions)] if isinstance(actions, int) else [int(a) for a in actions]
        active_index = int(prev.get("active", -1))
        prev_owned_team = int(prev.get("ball_owned_team", -1))
        prev_owned_player = int(prev.get("ball_owned_player", -1))
        next_owned_team = int(nxt.get("ball_owned_team", -1))
        next_owned_player = int(nxt.get("ball_owned_player", -1))
        prev_ball_x = self._ball_x(prev)
        next_ball_x = self._ball_x(nxt)

        if 0 <= active_index < len(action_list):
            active_action = action_list[active_index]
        else:
            active_action = None

        if (
            active_action in PASS_ACTIONS
            and prev_owned_team == 0
            and prev_owned_player == active_index
        ):
            self._pending_pass = {
                "from_player": active_index,
                "steps_left": max(1, self.reward_shaping.pending_pass_horizon),
            }

        if (
            active_action == SHOT_ACTION
            and prev_owned_team == 0
            and prev_owned_player == active_index
            and self.reward_shaping.shot_attempt_reward != 0.0
        ):
            reward_bonus += self.reward_shaping.shot_attempt_reward
            shaping_info["shot_attempt_bonus"] = self.reward_shaping.shot_attempt_reward

        if (
            prev_owned_team == 0
            and float(prev.get("ball", [0.0])[0]) >= self.reward_shaping.attacking_x_threshold
            and self.reward_shaping.attacking_possession_reward != 0.0
        ):
            reward_bonus += self.reward_shaping.attacking_possession_reward
            shaping_info["attacking_possession_bonus"] = self.reward_shaping.attacking_possession_reward

        if self._pending_pass is not None:
            self._pending_pass["steps_left"] -= 1
            pass_resolved = False
            from_player = int(self._pending_pass["from_player"])
            if next_owned_team == 0 and next_owned_player != -1 and next_owned_player != from_player:
                if self.reward_shaping.pass_success_reward != 0.0:
                    reward_bonus += self.reward_shaping.pass_success_reward
                    shaping_info["pass_success_bonus"] = self.reward_shaping.pass_success_reward
                if self.reward_shaping.pass_progress_reward_scale != 0.0:
                    progress_bonus = max(0.0, next_ball_x - prev_ball_x) * self.reward_shaping.pass_progress_reward_scale
                    if progress_bonus != 0.0:
                        reward_bonus += progress_bonus
                        shaping_info["pass_progress_bonus"] = progress_bonus
                pass_resolved = True
            elif next_owned_team == 1 or (done and next_owned_team != 0):
                if self.reward_shaping.pass_failure_penalty != 0.0:
                    reward_bonus -= self.reward_shaping.pass_failure_penalty
                    shaping_info["pass_failure_penalty"] = -self.reward_shaping.pass_failure_penalty
                pass_resolved = True
            elif self._pending_pass["steps_left"] <= 0:
                self._pending_pass = None
            if pass_resolved:
                self._pending_pass = None

        if (
            self.reward_shaping.final_third_entry_reward != 0.0
            and next_owned_team == 0
            and prev_ball_x < self.reward_shaping.attacking_x_threshold
            and next_ball_x >= self.reward_shaping.attacking_x_threshold
        ):
            reward_bonus += self.reward_shaping.final_third_entry_reward
            shaping_info["final_third_entry_bonus"] = self.reward_shaping.final_third_entry_reward

        if (
            self.reward_shaping.possession_retention_reward != 0.0
            and prev_owned_team == 0
            and next_owned_team == 0
        ):
            reward_bonus += self.reward_shaping.possession_retention_reward
            shaping_info["possession_retention_bonus"] = self.reward_shaping.possession_retention_reward

        if (
            self.reward_shaping.own_half_turnover_penalty != 0.0
            and prev_owned_team == 0
            and next_owned_team == 1
            and prev_ball_x <= self.reward_shaping.own_half_x_threshold
        ):
            reward_bonus -= self.reward_shaping.own_half_turnover_penalty
            shaping_info["own_half_turnover_penalty"] = -self.reward_shaping.own_half_turnover_penalty

        if (
            self.reward_shaping.possession_recovery_reward != 0.0
            and prev_owned_team != 0
            and next_owned_team == 0
        ):
            reward_bonus += self.reward_shaping.possession_recovery_reward
            shaping_info["possession_recovery_bonus"] = self.reward_shaping.possession_recovery_reward

        if (
            self.reward_shaping.defensive_third_recovery_reward != 0.0
            and prev_owned_team == 1
            and next_owned_team == 0
            and prev_ball_x <= self.reward_shaping.defensive_x_threshold
        ):
            reward_bonus += self.reward_shaping.defensive_third_recovery_reward
            shaping_info["defensive_third_recovery_bonus"] = self.reward_shaping.defensive_third_recovery_reward

        if (
            self.reward_shaping.opponent_attacking_possession_penalty != 0.0
            and next_owned_team == 1
            and next_ball_x <= self.reward_shaping.defensive_x_threshold
        ):
            reward_bonus -= self.reward_shaping.opponent_attacking_possession_penalty
            shaping_info["opponent_attacking_possession_penalty"] = -self.reward_shaping.opponent_attacking_possession_penalty

        return reward_bonus, shaping_info
