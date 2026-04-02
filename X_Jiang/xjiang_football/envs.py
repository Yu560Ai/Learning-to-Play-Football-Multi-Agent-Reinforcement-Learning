from __future__ import annotations

import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from xjiang_football.features import build_tactical_features, extract_feature_metrics
from xjiang_football.rewards import RewardShapingConfig, compute_shaped_reward, extract_step_metrics
from xjiang_football.utils import (
    NUM_TACTICAL_ACTIONS,
    ControlledRoleMap,
    build_controlled_role_map,
    tactical_mode_indices,
    policy_head_indices,
    tactical_action_mask,
    translate_tactical_actions,
)


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _football_source_root() -> Path:
    return _project_root() / "football-master"


def _ensure_engine_fonts(football_root: Path) -> None:
    engine_fonts = football_root / "third_party" / "gfootball_engine" / "fonts"
    bundled_fonts = football_root / "third_party" / "fonts"
    required_font = engine_fonts / "AlegreyaSansSC-ExtraBold.ttf"
    if required_font.exists() or not bundled_fonts.exists():
        return
    engine_fonts.mkdir(parents=True, exist_ok=True)
    for src in bundled_fonts.iterdir():
        dst = engine_fonts / src.name
        if src.is_file():
            shutil.copy2(src, dst)


def ensure_local_gfootball_path() -> Path:
    football_root = _football_source_root()
    _ensure_engine_fonts(football_root)
    for path in (football_root, football_root / "third_party"):
        path_str = str(path)
        if path.exists() and path_str not in sys.path:
            sys.path.insert(0, path_str)
    return football_root


def _bundled_football_env_site_packages() -> list[Path]:
    bundled_env_root = _football_source_root() / "football-env"
    return sorted(bundled_env_root.glob("lib/python*/site-packages"))


def _ensure_gym_compatibility() -> None:
    try:
        import gym  # type: ignore  # noqa: F401
        return
    except ImportError:
        pass

    for site_packages in _bundled_football_env_site_packages():
        site_packages_str = str(site_packages)
        if site_packages.exists() and site_packages_str not in sys.path:
            sys.path.insert(0, site_packages_str)
        try:
            import gym  # type: ignore  # noqa: F401
            return
        except ImportError:
            continue

    try:
        import gymnasium as gymnasium  # type: ignore
    except ImportError:
        return
    sys.modules.setdefault("gym", gymnasium)


def import_football_env():
    ensure_local_gfootball_path()
    _ensure_gym_compatibility()
    import gfootball.env as football_env  # type: ignore

    return football_env


@dataclass
class EnvSpec:
    obs_dim: int
    obs_shape: tuple[int, ...]
    action_dim: int
    num_players: int


class FootballEnvWrapper:
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
        reward_shaping: RewardShapingConfig | None = None,
    ) -> None:
        self._requested_num_players = num_controlled_players
        self.reward_shaping = reward_shaping or RewardShapingConfig()
        self.role_map = ControlledRoleMap(
            controlled_indices=np.zeros(0, dtype=np.int64),
            role_ids=np.zeros(0, dtype=np.int64),
            role_names=tuple(),
            goalkeeper_controlled=False,
        )

        config_options = dict(other_config_options or {})
        if write_video:
            config_options.setdefault("dump_full_episodes", True)

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
            other_config_options=config_options,
        )
        self.spec = self._bootstrap_spec()

    def _bootstrap_spec(self) -> EnvSpec:
        self.env.reset()
        raw_observation = self.get_raw_observation()
        self._sync_role_map(raw_observation)
        tactical_obs = build_tactical_features(raw_observation, role_map=self.role_map)
        obs_shape = tuple(int(dim) for dim in tactical_obs.shape[1:])
        obs_dim = int(np.prod(obs_shape))
        inferred_players = max(int(tactical_obs.shape[0]), min(self._requested_num_players, int(tactical_obs.shape[0] or 1)))
        return EnvSpec(
            obs_dim=obs_dim,
            obs_shape=obs_shape,
            action_dim=NUM_TACTICAL_ACTIONS,
            num_players=inferred_players,
        )

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
        self.env.reset()
        raw_observation = self.get_raw_observation()
        self._sync_role_map(raw_observation)
        return self._build_observation(raw_observation)

    def step(
        self,
        actions: np.ndarray | list[int] | int,
    ) -> tuple[np.ndarray, np.ndarray, bool, dict[str, Any]]:
        previous_raw = self.get_raw_observation()
        low_level_action = translate_tactical_actions(actions, previous_raw, self.role_map)
        _, reward, done, info = self.env.step(low_level_action)
        next_raw = self.get_raw_observation()
        self._sync_role_map(next_raw)

        formatted_obs = self._build_observation(next_raw)
        formatted_reward = self._format_reward(reward)
        tactical_actions = np.asarray(actions, dtype=np.int64).reshape(self.num_players)

        info_dict = dict(info or {})
        shaped_bonus, shaping_info = compute_shaped_reward(
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
        info_dict.update(extract_step_metrics(next_raw, role_map=self.role_map))
        info_dict.update(shaping_info)
        return formatted_obs, formatted_reward, bool(done), info_dict

    def close(self) -> None:
        if hasattr(self.env, "close"):
            self.env.close()

    def sample_random_action(self) -> np.ndarray | int:
        sample = np.random.randint(0, NUM_TACTICAL_ACTIONS, size=self.num_players, dtype=np.int64)
        return int(sample[0]) if self.num_players == 1 else sample

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

    def get_role_ids(self) -> np.ndarray:
        return self.role_map.role_ids.copy()

    def get_role_names(self) -> tuple[str, ...]:
        return self.role_map.role_names

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
        return np.asarray(features, dtype=np.float32).reshape(self.num_players, self.obs_dim)

    def _format_reward(self, reward: Any) -> np.ndarray:
        array = np.asarray(reward, dtype=np.float32)
        if array.ndim == 0:
            array = np.expand_dims(array, axis=0)
        if array.size == 1 and self.num_players > 1:
            array = np.repeat(array, self.num_players)
        return array.reshape(self.num_players)
