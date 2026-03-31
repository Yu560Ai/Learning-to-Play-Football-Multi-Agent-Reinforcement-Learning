from __future__ import annotations

import sys
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


def _project_root() -> Path:
    # X_Jiang/xjiang_football/envs.py -> project root
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

    candidate_paths = [
        football_root,
        football_root / "third_party",
    ]

    for path in candidate_paths:
        path_str = str(path)
        if path.exists() and path_str not in sys.path:
            sys.path.insert(0, path_str)

    return football_root


def import_football_env():
    ensure_local_gfootball_path()
    try:
        import gfootball.env as football_env  # type: ignore
    except ImportError as exc:
        football_root = _football_source_root()
        raise ImportError(
            "Could not import gfootball from local football-master. "
            f"Expected source tree at: {football_root}"
        ) from exc
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
    ) -> None:
        self._requested_num_players = num_controlled_players
        config_options = dict(other_config_options or {})
        if write_video:
            # GFootball only persists replay videos when a dump is also enabled.
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

        self.spec = self._infer_spec()

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

        return EnvSpec(
            obs_dim=obs_dim,
            obs_shape=obs_shape,
            action_dim=action_dim,
            num_players=num_players,
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
        observation = self.env.reset()
        return self._format_observation(observation)

    def step(
        self,
        actions: np.ndarray | list[int] | int,
    ) -> tuple[np.ndarray, np.ndarray, bool, dict[str, Any]]:
        action_input = self._format_action(actions)
        observation, reward, done, info = self.env.step(action_input)

        formatted_obs = self._format_observation(observation)
        formatted_reward = self._format_reward(reward)

        if info is None:
            info = {}

        return formatted_obs, formatted_reward, bool(done), dict(info)

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
