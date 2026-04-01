from __future__ import annotations

import multiprocessing as mp
import sys
import traceback
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


def _build_env_kwargs(
    base_kwargs: dict[str, Any],
    env_index: int,
    *,
    num_envs: int,
    base_seed: int | None,
) -> dict[str, Any]:
    env_kwargs = dict(base_kwargs)
    other_config_options = dict(env_kwargs.get("other_config_options") or {})

    if base_seed is not None and num_envs > 1:
        other_config_options.setdefault("game_engine_random_seed", int(base_seed) + env_index)

    env_kwargs["other_config_options"] = other_config_options

    logdir = env_kwargs.get("logdir")
    if logdir and num_envs > 1:
        env_logdir = Path(logdir) / f"env_{env_index:02d}"
        env_logdir.mkdir(parents=True, exist_ok=True)
        env_kwargs["logdir"] = str(env_logdir)

    return env_kwargs


def _auto_reset_if_done(
    env: FootballEnvWrapper,
    observation: np.ndarray,
    reward: np.ndarray,
    done: bool,
    info: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, bool, dict[str, Any]]:
    info_dict = dict(info or {})
    if not done:
        return observation, reward, False, info_dict

    score = env.get_score()
    if score is not None:
        info_dict["final_score"] = score

    reset_observation = env.reset()
    info_dict["auto_reset"] = True
    return reset_observation, reward, True, info_dict


def _subproc_worker(connection: Any, env_kwargs: dict[str, Any]) -> None:
    env: FootballEnvWrapper | None = None
    try:
        env = FootballEnvWrapper(**env_kwargs)
        connection.send(("ready", env.spec))
        while True:
            command, payload = connection.recv()
            if command == "reset":
                connection.send(("result", env.reset()))
            elif command == "step":
                step_result = env.step(payload)
                connection.send(("result", _auto_reset_if_done(env, *step_result)))
            elif command == "close":
                connection.send(("result", None))
                break
            else:
                raise ValueError(f"Unsupported worker command: {command!r}")
    except EOFError:
        pass
    except Exception:
        try:
            connection.send(("error", traceback.format_exc()))
        except (BrokenPipeError, EOFError):
            pass
    finally:
        if env is not None:
            env.close()
        connection.close()


class FootballVecEnv:
    def __init__(
        self,
        *,
        num_envs: int = 1,
        base_seed: int | None = None,
        **env_kwargs: Any,
    ) -> None:
        if num_envs < 1:
            raise ValueError("num_envs must be at least 1.")
        if num_envs > 1 and env_kwargs.get("render", False):
            raise ValueError("Parallel rollout does not support render=True. Use --num-envs 1 when rendering.")

        self.num_envs = int(num_envs)
        self._closed = False
        self._use_subprocesses = self.num_envs > 1
        self._envs: list[FootballEnvWrapper] = []
        self._connections: list[Any] = []
        self._processes: list[mp.Process] = []

        if self._use_subprocesses:
            ctx = mp.get_context("spawn")
            reference_spec: EnvSpec | None = None
            for env_index in range(self.num_envs):
                parent_conn, child_conn = ctx.Pipe()
                worker_env_kwargs = _build_env_kwargs(
                    env_kwargs,
                    env_index,
                    num_envs=self.num_envs,
                    base_seed=base_seed,
                )
                process = ctx.Process(
                    target=_subproc_worker,
                    args=(child_conn, worker_env_kwargs),
                    daemon=True,
                )
                process.start()
                child_conn.close()

                try:
                    message_type, payload = parent_conn.recv()
                except EOFError as exc:
                    raise RuntimeError("FootballVecEnv worker exited before initialization completed.") from exc
                if message_type == "error":
                    raise RuntimeError(f"FootballVecEnv worker failed to start:\n{payload}")

                worker_spec = payload
                if reference_spec is None:
                    reference_spec = worker_spec
                elif worker_spec != reference_spec:
                    raise ValueError(
                        "All vectorized football environments must expose the same observation/action spec."
                    )

                self._connections.append(parent_conn)
                self._processes.append(process)

            if reference_spec is None:
                raise RuntimeError("Failed to initialize vectorized football environments.")
            self.spec = reference_spec
        else:
            self._envs.append(
                FootballEnvWrapper(
                    **_build_env_kwargs(
                        env_kwargs,
                        0,
                        num_envs=self.num_envs,
                        base_seed=base_seed,
                    )
                )
            )
            self.spec = self._envs[0].spec

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

    def _recv(self, connection: Any) -> Any:
        message_type, payload = connection.recv()
        if message_type == "error":
            raise RuntimeError(f"FootballVecEnv worker failed:\n{payload}")
        return payload

    def _split_actions(self, actions: np.ndarray | list[int] | int) -> list[np.ndarray | int]:
        if self.num_envs == 1:
            action_array = np.asarray(actions, dtype=np.int64)
            if self.num_players == 1:
                return [int(action_array.reshape(-1)[0])]
            return [action_array.reshape(self.num_players)]

        action_array = np.asarray(actions, dtype=np.int64)
        if self.num_players == 1:
            if action_array.shape == (self.num_envs,):
                return [int(action) for action in action_array.tolist()]
            return [int(action[0]) for action in action_array.reshape(self.num_envs, 1)]

        reshaped = action_array.reshape(self.num_envs, self.num_players)
        return [reshaped[env_index] for env_index in range(self.num_envs)]

    def reset(self) -> np.ndarray:
        if self._use_subprocesses:
            for connection in self._connections:
                connection.send(("reset", None))
            observations = [self._recv(connection) for connection in self._connections]
        else:
            observations = [env.reset() for env in self._envs]
        return np.stack(observations, axis=0)

    def step(
        self,
        actions: np.ndarray | list[int] | int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]]]:
        split_actions = self._split_actions(actions)
        if self._use_subprocesses:
            for connection, env_actions in zip(self._connections, split_actions, strict=True):
                connection.send(("step", env_actions))
            results = [self._recv(connection) for connection in self._connections]
        else:
            results = []
            for env, env_actions in zip(self._envs, split_actions, strict=True):
                results.append(_auto_reset_if_done(env, *env.step(env_actions)))

        observations, rewards, dones, infos = zip(*results, strict=True)
        return (
            np.stack(observations, axis=0),
            np.stack(rewards, axis=0),
            np.asarray(dones, dtype=np.bool_),
            list(infos),
        )

    def close(self) -> None:
        if self._closed:
            return

        if self._use_subprocesses:
            for connection in self._connections:
                try:
                    connection.send(("close", None))
                except (BrokenPipeError, EOFError):
                    pass
            for connection in self._connections:
                try:
                    self._recv(connection)
                except (EOFError, BrokenPipeError):
                    pass
                connection.close()
            for process in self._processes:
                process.join(timeout=1.0)
        else:
            for env in self._envs:
                env.close()

        self._closed = True
