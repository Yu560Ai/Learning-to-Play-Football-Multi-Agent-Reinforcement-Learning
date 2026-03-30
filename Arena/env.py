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
        render: bool = False,
        save_video: bool = False,
        video_dir: str = "Arena/videos",
    ) -> None:
        football_env = import_football_env()
        if save_video:
            Path(video_dir).mkdir(parents=True, exist_ok=True)
        self.env = football_env.create_environment(
            env_name=env_name,
            representation="raw",
            rewards=rewards,
            render=render,
            write_video=save_video,
            write_full_episode_dumps=save_video,
            logdir=video_dir if save_video else "",
            number_of_left_players_agent_controls=1,
            number_of_right_players_agent_controls=1,
        )
        action_space = self.env.action_space
        if hasattr(action_space, "nvec"):
            self.num_actions = int(action_space.nvec[0])
        else:
            self.num_actions = int(action_space.n)
        self._last_observation: list[dict[str, Any]] | None = None

    def reset(self) -> tuple[dict[str, Any], dict[str, Any]]:
        observations = self.env.reset()
        self._last_observation = observations
        return observations[0], observations[1]

    def step(self, left_action: int, right_action: int) -> tuple[dict[str, Any], dict[str, Any], np.ndarray, bool, dict[str, Any]]:
        observations, rewards, done, info = self.env.step([left_action, right_action])
        self._last_observation = observations
        return observations[0], observations[1], np.asarray(rewards, dtype=np.float32), bool(done), dict(info)

    def get_score(self) -> tuple[int, int]:
        if not self._last_observation:
            raise RuntimeError("reset() must be called before get_score().")
        score = self._last_observation[0]["score"]
        return int(score[0]), int(score[1])

    def close(self) -> None:
        if hasattr(self.env, "close"):
            self.env.close()
