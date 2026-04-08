from __future__ import annotations

import shutil
import sys
import multiprocessing as mp
from dataclasses import dataclass
from multiprocessing.connection import Connection
from pathlib import Path
from typing import Any

import numpy as np

from xjiang_football.features import ENGINEERED_FEATURE_DIM, build_engineered_features
from xjiang_football.rewards import RewardShapingConfig, compute_shaped_reward, _checkpoint_value


REDUCED_ACTION_INDICES: tuple[int, ...] = (
    0,
    1, 2, 3, 4, 5, 6, 7, 8,
    9, 10, 11,
    12,
    13, 14, 15,
    17, 18,
)

REDUCED_ACTION_NAMES: tuple[str, ...] = (
    "idle",
    "left",
    "top_left",
    "top",
    "top_right",
    "right",
    "bottom_right",
    "bottom",
    "bottom_left",
    "long_pass",
    "high_pass",
    "short_pass",
    "shot",
    "sprint",
    "release_direction",
    "release_sprint",
    "dribble",
    "release_dribble",
)

ACADEMY_ACTION_INDICES: tuple[int, ...] = (
    0,
    1, 2, 3, 4, 5, 6, 7, 8,
    11,
    12,
    13, 14, 15,
    17, 18,
)

ACADEMY_ACTION_NAMES: tuple[str, ...] = (
    "idle",
    "left",
    "top_left",
    "top",
    "top_right",
    "right",
    "bottom_right",
    "bottom",
    "bottom_left",
    "short_pass",
    "shot",
    "sprint",
    "release_direction",
    "release_sprint",
    "dribble",
    "release_dribble",
)

SOLO_ACADEMY_ACTION_INDICES: tuple[int, ...] = (
    0,
    1, 2, 3, 4, 5, 6, 7, 8,
    12,
    13, 14, 15,
    17, 18,
)

SOLO_ACADEMY_ACTION_NAMES: tuple[str, ...] = (
    "idle",
    "left",
    "top_left",
    "top",
    "top_right",
    "right",
    "bottom_right",
    "bottom",
    "bottom_left",
    "shot",
    "sprint",
    "release_direction",
    "release_sprint",
    "dribble",
    "release_dribble",
)

SOLO_MINIMAL_ACTION_INDICES: tuple[int, ...] = (
    0,
    1, 2, 3, 4, 5, 6, 7, 8,
    12,
)

SOLO_MINIMAL_ACTION_NAMES: tuple[str, ...] = (
    "idle",
    "left",
    "top_left",
    "top",
    "top_right",
    "right",
    "bottom_right",
    "bottom",
    "bottom_left",
    "shot",
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
        representation: str = "simple115v2",
        rewards: str = "scoring",
        stacked: bool = False,
        render: bool = False,
        write_video: bool = False,
        logdir: str | None = None,
        num_controlled_players: int = 2,
        channel_dimensions: tuple[int, int] = (42, 42),
        other_config_options: dict[str, Any] | None = None,
        reward_shaping: RewardShapingConfig | None = None,
        use_engineered_features: bool = True,
        collect_feature_metrics: bool = False,
        action_set: str = "full",
        force_shoot_in_zone: bool = False,
        force_shoot_x_threshold: float = 0.3,
        force_shoot_y_threshold: float = 0.24,
    ) -> None:
        self.reward_shaping = reward_shaping or RewardShapingConfig()
        self.use_engineered_features = bool(use_engineered_features)
        self.collect_feature_metrics = bool(collect_feature_metrics)
        self.force_shoot_in_zone = bool(force_shoot_in_zone)
        self.force_shoot_x_threshold = float(force_shoot_x_threshold)
        self.force_shoot_y_threshold = float(force_shoot_y_threshold)
        self._possession_checkpoint_value: float | None = None
        if action_set == "solo_minimal":
            self.action_map = np.asarray(SOLO_MINIMAL_ACTION_INDICES, dtype=np.int64)
            self.action_names = SOLO_MINIMAL_ACTION_NAMES
        elif action_set == "solo_academy":
            self.action_map = np.asarray(SOLO_ACADEMY_ACTION_INDICES, dtype=np.int64)
            self.action_names = SOLO_ACADEMY_ACTION_NAMES
        elif action_set == "academy":
            self.action_map = np.asarray(ACADEMY_ACTION_INDICES, dtype=np.int64)
            self.action_names = ACADEMY_ACTION_NAMES
        else:
            self.action_map = np.asarray(REDUCED_ACTION_INDICES, dtype=np.int64)
            self.action_names = REDUCED_ACTION_NAMES
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
        observation = self.reset()
        obs_shape = tuple(int(dim) for dim in observation.shape[1:])
        obs_dim = int(np.prod(obs_shape))
        return EnvSpec(
            obs_dim=obs_dim,
            obs_shape=obs_shape,
            action_dim=int(self.action_map.shape[0]),
            num_players=int(observation.shape[0]),
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
        self._possession_checkpoint_value = None
        observation = self.env.reset()
        return self._augment_observation(observation, raw_obs=self.get_raw_observation())

    def step(
        self,
        action: int | np.ndarray | list[int],
    ) -> tuple[np.ndarray, np.ndarray, bool, dict[str, Any]]:
        previous_raw = self.get_raw_observation()
        action_indices = self._normalize_actions(action)
        mapped_action = self.action_map[action_indices]
        forced_shots = self._apply_shoot_assist(previous_raw, mapped_action)
        observation, reward, done, info = self.env.step(mapped_action)
        next_raw = self.get_raw_observation()
        done = bool(done or self._is_terminal_from_raw(next_raw))
        augmented_observation = self._augment_observation(observation, raw_obs=next_raw)
        checkpoint_progress = self._compute_checkpoint_progress(previous_raw, next_raw)
        shaped_reward, shaping_metrics = compute_shaped_reward(
            previous_raw_observation=previous_raw,
            next_raw_observation=next_raw,
            mapped_action=mapped_action,
            config=self.reward_shaping,
            num_players=self.num_players,
            checkpoint_progress_override=checkpoint_progress,
        )
        reward_array = np.asarray(reward, dtype=np.float32).reshape(self.num_players)
        reward_array = reward_array + shaped_reward
        info_dict = dict(info or {})
        if self.collect_feature_metrics:
            from xjiang_football.features import extract_feature_metrics

            info_dict.update(extract_feature_metrics(next_raw))
        info_dict.update(shaping_metrics)
        info_dict["forced_shot_overrides"] = float(forced_shots)
        info_dict["steps_left"] = self._mean_steps_left(next_raw)
        return augmented_observation, reward_array, done, info_dict

    def close(self) -> None:
        if hasattr(self.env, "close"):
            self.env.close()

    def sample_random_action(self) -> np.ndarray:
        return np.random.randint(self.action_dim, size=self.num_players, dtype=np.int64)

    def get_raw_observation(self) -> Any:
        base_env = getattr(self.env, "unwrapped", self.env)
        if hasattr(base_env, "observation"):
            return base_env.observation()
        if hasattr(self.env, "observation"):
            return self.env.observation()
        return None

    def get_score(self) -> tuple[int, int] | None:
        raw_observation = self.get_raw_observation()
        raw_list = raw_observation if isinstance(raw_observation, list) else [raw_observation]
        if raw_list and isinstance(raw_list[0], dict) and "score" in raw_list[0]:
            left_score, right_score = raw_list[0]["score"]
            return int(left_score), int(right_score)
        return None

    def action_name(self, action_id: int) -> str:
        if 0 <= action_id < len(self.action_names):
            return self.action_names[action_id]
        return f"action_{action_id}"

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
        base_array = np.asarray(observation, dtype=np.float32).reshape(-1, np.asarray(observation).shape[-1])
        if not self.use_engineered_features:
            return base_array
        raw = raw_obs if raw_obs is not None else self.get_raw_observation()
        raw_list = raw if isinstance(raw, list) else [raw]
        engineered_features: list[np.ndarray] = []
        for idx in range(base_array.shape[0]):
            raw_single = raw_list[idx] if idx < len(raw_list) else raw_list[0]
            engineered_features.append(build_engineered_features(raw_single if isinstance(raw_single, dict) else {}))
        return np.concatenate([base_array, np.stack(engineered_features, axis=0)], axis=1)

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

    def _compute_checkpoint_progress(self, previous_raw_observation: Any, next_raw_observation: Any) -> float:
        prev_list = previous_raw_observation if isinstance(previous_raw_observation, list) else [previous_raw_observation]
        next_list = next_raw_observation if isinstance(next_raw_observation, list) else [next_raw_observation]
        if not prev_list or not next_list:
            self._possession_checkpoint_value = None
            return 0.0

        previous = prev_list[0] if isinstance(prev_list[0], dict) else {}
        current = next_list[0] if isinstance(next_list[0], dict) else {}
        prev_team = int(previous.get("ball_owned_team", -1))
        next_team = int(current.get("ball_owned_team", -1))
        if next_team != 0:
            self._possession_checkpoint_value = None
            return 0.0

        next_ball_x = float(np.asarray(current.get("ball", np.zeros(3)), dtype=np.float32)[0])
        next_value = _checkpoint_value(next_ball_x)

        if prev_team != 0 or self._possession_checkpoint_value is None:
            prev_ball_x = float(np.asarray(previous.get("ball", np.zeros(3)), dtype=np.float32)[0])
            baseline_value = _checkpoint_value(prev_ball_x) if prev_team == 0 else 0.0
            progress = max(0.0, next_value - baseline_value)
            self._possession_checkpoint_value = max(baseline_value, next_value)
            return float(progress)

        progress = max(0.0, next_value - self._possession_checkpoint_value)
        self._possession_checkpoint_value = max(self._possession_checkpoint_value, next_value)
        return float(progress)

    def _apply_shoot_assist(self, raw_observation: Any, mapped_action: np.ndarray) -> int:
        if not self.force_shoot_in_zone or self.num_players != 1 or mapped_action.size == 0:
            return 0
        raw_list = raw_observation if isinstance(raw_observation, list) else [raw_observation]
        raw = raw_list[0] if raw_list and isinstance(raw_list[0], dict) else {}
        owner_team = int(raw.get("ball_owned_team", -1))
        owner_player = int(raw.get("ball_owned_player", -1))
        active = int(raw.get("active", -1))
        ball = np.asarray(raw.get("ball", np.zeros(3)), dtype=np.float32)
        ball_x = float(ball[0]) if ball.size else 0.0
        ball_y = float(ball[1]) if ball.size > 1 else 0.0
        has_ball = owner_team == 0 and owner_player == active
        in_shoot_lane = ball_x >= self.force_shoot_x_threshold and abs(ball_y) <= self.force_shoot_y_threshold
        if has_ball and in_shoot_lane and int(mapped_action[0]) != 12:
            mapped_action[0] = 12
            return 1
        return 0


def _parallel_env_worker(conn: Connection, kwargs: dict[str, Any]) -> None:
    env = FootballEnvWrapper(**kwargs)
    try:
        while True:
            command, payload = conn.recv()
            if command == "reset":
                conn.send(env.reset())
            elif command == "step":
                conn.send(env.step(payload))
            elif command == "get_score":
                conn.send(env.get_score())
            elif command == "close":
                env.close()
                conn.close()
                break
            else:
                raise ValueError(f"Unknown command: {command}")
    finally:
        try:
            env.close()
        except Exception:
            pass


class ParallelFootballEnvWrapper:
    def __init__(self, num_envs: int = 1, **env_kwargs: Any) -> None:
        self.num_envs = int(num_envs)
        if self.num_envs < 1:
            raise ValueError("num_envs must be at least 1.")
        self._env_kwargs = dict(env_kwargs)
        self._local_env: FootballEnvWrapper | None = None
        self._processes: list[mp.Process] = []
        self._parents: list[Connection] = []

        if self.num_envs == 1:
            self._local_env = FootballEnvWrapper(**self._env_kwargs)
            self.spec = self._local_env.spec
        else:
            start_method = "fork" if "fork" in mp.get_all_start_methods() else "spawn"
            ctx = mp.get_context(start_method)
            for _ in range(self.num_envs):
                parent_conn, child_conn = ctx.Pipe()
                process = ctx.Process(target=_parallel_env_worker, args=(child_conn, self._env_kwargs), daemon=True)
                process.start()
                child_conn.close()
                self._parents.append(parent_conn)
                self._processes.append(process)
            first_obs = self.reset()
            obs_shape = tuple(int(dim) for dim in first_obs.shape[2:])
            self.spec = EnvSpec(
                obs_dim=int(np.prod(obs_shape)),
                obs_shape=obs_shape,
                action_dim=len(self.action_names),
                num_players=int(first_obs.shape[1]),
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

    @property
    def reward_shaping(self) -> RewardShapingConfig:
        if self._local_env is not None:
            return self._local_env.reward_shaping
        reward_shaping = self._env_kwargs.get("reward_shaping")
        if isinstance(reward_shaping, RewardShapingConfig):
            return reward_shaping
        return RewardShapingConfig()

    @property
    def action_names(self) -> tuple[str, ...]:
        if self._local_env is not None:
            return self._local_env.action_names
        action_set = self._env_kwargs.get("action_set", "full")
        if action_set == "solo_minimal":
            return SOLO_MINIMAL_ACTION_NAMES
        if action_set == "solo_academy":
            return SOLO_ACADEMY_ACTION_NAMES
        if action_set == "academy":
            return ACADEMY_ACTION_NAMES
        return REDUCED_ACTION_NAMES

    def action_name(self, action_id: int) -> str:
        names = self.action_names
        if 0 <= action_id < len(names):
            return names[action_id]
        return f"action_{action_id}"

    def reset(self) -> np.ndarray:
        if self._local_env is not None:
            return self._local_env.reset()[None, ...]
        for parent in self._parents:
            parent.send(("reset", None))
        observations = [parent.recv() for parent in self._parents]
        return np.stack(observations, axis=0).astype(np.float32)

    def step(self, actions: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]]]:
        action_array = np.asarray(actions, dtype=np.int64)
        if self.num_envs == 1:
            assert self._local_env is not None
            obs, reward, done, info = self._local_env.step(action_array.reshape(-1))
            info_dict = dict(info or {})
            if done:
                info_dict["_episode_score"] = self._local_env.get_score()
                obs = self._local_env.reset()
            return obs[None, ...], np.asarray(reward, dtype=np.float32)[None, ...], np.asarray([done], dtype=bool), [info_dict]

        for env_index, parent in enumerate(self._parents):
            parent.send(("step", action_array[env_index].reshape(-1)))

        observations: list[np.ndarray] = []
        rewards: list[np.ndarray] = []
        dones: list[bool] = []
        infos: list[dict[str, Any]] = []
        for env_index, parent in enumerate(self._parents):
            obs, reward, done, info = parent.recv()
            info_dict = dict(info or {})
            done_flag = bool(done)
            if done_flag:
                parent.send(("get_score", None))
                info_dict["_episode_score"] = parent.recv()
                parent.send(("reset", None))
                obs = parent.recv()
            observations.append(np.asarray(obs, dtype=np.float32))
            rewards.append(np.asarray(reward, dtype=np.float32))
            dones.append(done_flag)
            infos.append(info_dict)

        return (
            np.stack(observations, axis=0),
            np.stack(rewards, axis=0),
            np.asarray(dones, dtype=bool),
            infos,
        )

    def close(self) -> None:
        if self._local_env is not None:
            self._local_env.close()
            return
        for parent in self._parents:
            try:
                parent.send(("close", None))
            except Exception:
                pass
        for process in self._processes:
            process.join(timeout=2.0)
