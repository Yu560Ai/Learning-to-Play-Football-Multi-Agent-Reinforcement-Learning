from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from Y_Fu.yfu_football.envs import import_football_env


class ArenaMatchEnv:
    def __init__(
        self,
        env_name: str = "11_vs_11_kaggle",
        *,
        rewards: str = "scoring",
        representation: str = "raw",
        action_set: str = "default",
        left_players: int = 1,
        right_players: int = 1,
        channel_dimensions: tuple[int, int] = (42, 42),
        env_seed: int | None = None,
        render: bool = False,
        save_video: bool = False,
        video_dir: str = "Arena/videos",
    ) -> None:
        football_env = import_football_env()
        if save_video:
            Path(video_dir).mkdir(parents=True, exist_ok=True)
        other_config_options = {"action_set": action_set}
        if env_seed is not None:
            other_config_options["game_engine_random_seed"] = int(env_seed)
        self.env = football_env.create_environment(
            env_name=env_name,
            representation=representation,
            rewards=rewards,
            render=render,
            write_video=save_video,
            write_full_episode_dumps=save_video,
            logdir=video_dir if save_video else "",
            number_of_left_players_agent_controls=left_players,
            number_of_right_players_agent_controls=right_players,
            channel_dimensions=channel_dimensions,
            other_config_options=other_config_options,
        )
        self.left_players = left_players
        self.right_players = right_players
        self.representation = representation
        action_space = self.env.action_space
        if hasattr(action_space, "nvec"):
            self.num_actions = int(action_space.nvec[0])
        else:
            self.num_actions = int(action_space.n)
        self._last_observation: list[dict[str, Any]] | None = None

    def _split_observations(self, observations: Any) -> tuple[Any, Any]:
        if isinstance(observations, list):
            return observations[: self.left_players], observations[self.left_players : self.left_players + self.right_players]
        array = np.asarray(observations)
        return array[: self.left_players], array[self.left_players : self.left_players + self.right_players]

    def _combine_actions(self, left_action: int | list[int], right_action: int | list[int]) -> list[int]:
        def _normalize(action: int | list[int], expected: int) -> list[int]:
            if isinstance(action, list):
                return [int(a) for a in action]
            array = np.asarray(action, dtype=np.int64).reshape(-1)
            if array.size == 1 and expected == 1:
                return [int(array[0])]
            if array.size != expected:
                raise ValueError(f"Expected {expected} actions, got {array.size}.")
            return array.astype(int).tolist()

        return _normalize(left_action, self.left_players) + _normalize(right_action, self.right_players)

    def reset(self) -> tuple[Any, Any]:
        observations = self.env.reset()
        self._last_observation = observations
        return self._split_observations(observations)

    def step(self, left_action: int | list[int], right_action: int | list[int]) -> tuple[Any, Any, np.ndarray, bool, dict[str, Any]]:
        observations, rewards, done, info = self.env.step(self._combine_actions(left_action, right_action))
        self._last_observation = observations
        reward_array = np.asarray(rewards, dtype=np.float32).reshape(self.left_players + self.right_players)
        left_reward = float(np.mean(reward_array[: self.left_players]))
        right_reward = float(np.mean(reward_array[self.left_players : self.left_players + self.right_players]))
        left_obs, right_obs = self._split_observations(observations)
        return left_obs, right_obs, np.asarray([left_reward, right_reward], dtype=np.float32), bool(done), dict(info)

    def get_score(self) -> tuple[int, int]:
        base_env = getattr(self.env, "unwrapped", self.env)
        if hasattr(base_env, "observation"):
            raw_observation = base_env.observation()
        elif hasattr(self.env, "observation"):
            raw_observation = self.env.observation()
        else:
            raise RuntimeError("Could not access raw observation for score lookup.")
        if not isinstance(raw_observation, list) or not raw_observation:
            raise RuntimeError("reset() must be called before get_score().")
        score = raw_observation[0]["score"]
        return int(score[0]), int(score[1])

    def close(self) -> None:
        if hasattr(self.env, "close"):
            self.env.close()
