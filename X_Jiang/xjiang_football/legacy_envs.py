from __future__ import annotations

from typing import Any

import numpy as np

from xjiang_football.envs import import_football_env
from xjiang_football.legacy_features import build_tactical_features, extract_feature_metrics
from xjiang_football.legacy_rewards import LegacyRewardShapingConfig, compute_legacy_shaped_reward
from xjiang_football.utils import (
    NUM_TACTICAL_ACTIONS,
    ControlledRoleMap,
    build_controlled_role_map,
    tactical_mode_indices,
    policy_head_indices,
    tactical_action_mask,
    translate_tactical_actions,
)


class LegacyFootballEnvWrapper:
    def __init__(
        self,
        env_name: str = "5_vs_5",
        representation: str = "extracted",
        rewards: str = "scoring,checkpoints",
        stacked: bool = False,
        render: bool = False,
        write_video: bool = False,
        logdir: str | None = None,
        num_controlled_players: int = 4,
        channel_dimensions: tuple[int, int] = (42, 42),
        other_config_options: dict[str, Any] | None = None,
        reward_shaping: LegacyRewardShapingConfig | None = None,
    ) -> None:
        self._requested_num_players = num_controlled_players
        self.reward_shaping = reward_shaping or LegacyRewardShapingConfig()
        self.role_map = ControlledRoleMap(
            controlled_indices=np.zeros(0, dtype=np.int64),
            role_ids=np.zeros(0, dtype=np.int64),
            role_names=tuple(),
            goalkeeper_controlled=False,
        )
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
            other_config_options=dict(other_config_options or {}),
        )
        self.reset()

    @property
    def obs_dim(self) -> int:
        return int(self.reset().shape[-1])

    @property
    def action_dim(self) -> int:
        return NUM_TACTICAL_ACTIONS

    @property
    def num_players(self) -> int:
        return int(max(1, self.reset().shape[0]))

    @property
    def obs_shape(self) -> tuple[int, ...]:
        obs = self.reset()
        return tuple(int(dim) for dim in obs.shape[1:])

    def reset(self) -> np.ndarray:
        self.env.reset()
        raw_observation = self.get_raw_observation()
        self._sync_role_map(raw_observation)
        return self._build_observation(raw_observation)

    def step(self, actions: np.ndarray | list[int] | int) -> tuple[np.ndarray, np.ndarray, bool, dict[str, Any]]:
        previous_raw = self.get_raw_observation()
        low_level_action = translate_tactical_actions(actions, previous_raw, self.role_map)
        _, reward, done, info = self.env.step(low_level_action)
        next_raw = self.get_raw_observation()
        self._sync_role_map(next_raw)

        formatted_obs = self._build_observation(next_raw)
        formatted_reward = self._format_reward(reward)
        tactical_actions = np.asarray(actions, dtype=np.int64).reshape(self.num_players)

        info_dict = dict(info or {})
        shaped_bonus, shaping_info = compute_legacy_shaped_reward(
            previous_raw_observation=previous_raw,
            next_raw_observation=next_raw,
            tactical_actions=tactical_actions,
            role_map=self.role_map,
            config=self.reward_shaping,
        )
        if shaped_bonus.size > 0:
            formatted_reward = formatted_reward + shaped_bonus.astype(np.float32)

        info_dict["reward_bonus"] = float(np.mean(shaped_bonus)) if shaped_bonus.size > 0 else 0.0
        info_dict["controlled_role_names"] = list(self.role_map.role_names)
        info_dict["controlled_indices"] = self.role_map.controlled_indices.tolist()
        info_dict["tactical_actions"] = tactical_actions.tolist()
        info_dict.update(extract_feature_metrics(next_raw, role_map=self.role_map))
        info_dict.update(shaping_info)
        return formatted_obs, formatted_reward, bool(done), info_dict

    def close(self) -> None:
        if hasattr(self.env, "close"):
            self.env.close()

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

    def get_action_mask(self, raw_observation: Any | None = None) -> np.ndarray:
        raw = self.get_raw_observation() if raw_observation is None else raw_observation
        return tactical_action_mask(raw, self.role_map)

    def get_policy_head_indices(self, raw_observation: Any | None = None) -> np.ndarray:
        raw = self.get_raw_observation() if raw_observation is None else raw_observation
        return policy_head_indices(raw, self.role_map)

    def get_tactical_mode_indices(self, raw_observation: Any | None = None) -> np.ndarray:
        raw = self.get_raw_observation() if raw_observation is None else raw_observation
        return tactical_mode_indices(raw, self.role_map)

    def _sync_role_map(self, raw_observation: Any) -> None:
        self.role_map = build_controlled_role_map(raw_observation, requested_players=self._requested_num_players)

    def _build_observation(self, raw_observation: Any) -> np.ndarray:
        features = build_tactical_features(raw_observation, role_map=self.role_map)
        return np.asarray(features, dtype=np.float32).reshape(features.shape[0], -1)

    def _format_reward(self, reward: Any) -> np.ndarray:
        array = np.asarray(reward, dtype=np.float32)
        if array.ndim == 0:
            array = np.expand_dims(array, axis=0)
        if array.size == 1 and self._requested_num_players > 1:
            array = np.repeat(array, self._requested_num_players)
        return array.reshape(-1)

