from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch


@dataclass(frozen=True)
class ChunkRecord:
    path: Path
    num_rows: int


class OfflineGRFDataset:
    def __init__(
        self,
        data_dirs: list[str | Path],
        *,
        reward_key: str = "reward",
        chunk_reuse_batches: int = 32,
    ) -> None:
        if reward_key not in {"reward", "score_reward"}:
            raise ValueError("reward_key must be either 'reward' or 'score_reward'.")
        self.reward_key = reward_key
        self.chunk_reuse_batches = max(1, int(chunk_reuse_batches))
        self.data_dirs = [Path(path) for path in data_dirs]
        if not self.data_dirs:
            raise ValueError("At least one dataset directory is required.")

        self.chunk_records: list[ChunkRecord] = []
        self.manifests: list[dict[str, Any]] = []
        self.obs_dim = 0
        self.action_dim = 0
        self.obs_shape: tuple[int, ...] = ()
        self.num_controlled_players = 0
        self.env_config: dict[str, Any] = {}

        for directory in self.data_dirs:
            manifest_path = directory / "manifest.json"
            if not manifest_path.exists():
                raise FileNotFoundError(f"Offline dataset manifest not found: {manifest_path}")
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.manifests.append(manifest)
            if not self.env_config:
                self.obs_dim = int(manifest["obs_dim"])
                self.action_dim = int(manifest["action_dim"])
                self.obs_shape = tuple(int(dim) for dim in manifest["obs_shape"])
                self.num_controlled_players = int(manifest["num_controlled_players"])
                self.env_config = {
                    "env_name": manifest["env_name"],
                    "representation": manifest["representation"],
                    "rewards": manifest["rewards"],
                    "num_controlled_players": manifest["num_controlled_players"],
                    "channel_dimensions": manifest["channel_dimensions"],
                    "reward_shaping": dict(manifest.get("reward_shaping", {})),
                }
            else:
                self._validate_manifest_compatibility(directory, manifest)

            for chunk_metadata in manifest["chunks"]:
                self.chunk_records.append(
                    ChunkRecord(
                        path=directory / chunk_metadata["filename"],
                        num_rows=int(chunk_metadata["num_rows"]),
                    )
                )

        if not self.chunk_records:
            raise ValueError("No chunk files were found in the supplied dataset directories.")

        chunk_weights = np.asarray([record.num_rows for record in self.chunk_records], dtype=np.float64)
        self.chunk_sampling_probs = chunk_weights / max(float(chunk_weights.sum()), 1.0)
        self._active_chunk_index: int | None = None
        self._active_chunk: dict[str, np.ndarray] | None = None
        self._active_valid_indices: np.ndarray | None = None
        self._active_terminal_mask: np.ndarray | None = None
        self._remaining_batches_on_active_chunk = 0

    def _validate_manifest_compatibility(self, directory: Path, manifest: dict[str, Any]) -> None:
        checks = {
            "obs_dim": int(manifest["obs_dim"]) == self.obs_dim,
            "action_dim": int(manifest["action_dim"]) == self.action_dim,
            "obs_shape": tuple(int(dim) for dim in manifest["obs_shape"]) == self.obs_shape,
            "env_name": str(manifest["env_name"]) == str(self.env_config["env_name"]),
            "representation": str(manifest["representation"]) == str(self.env_config["representation"]),
            "rewards": str(manifest["rewards"]) == str(self.env_config["rewards"]),
            "num_controlled_players": int(manifest["num_controlled_players"]) == self.num_controlled_players,
            "channel_dimensions": tuple(int(dim) for dim in manifest["channel_dimensions"])
            == tuple(int(dim) for dim in self.env_config["channel_dimensions"]),
            "reward_shaping": dict(manifest.get("reward_shaping", {})) == dict(self.env_config.get("reward_shaping", {})),
        }
        mismatched = [name for name, matches in checks.items() if not matches]
        if mismatched:
            mismatch_text = ", ".join(mismatched)
            raise ValueError(
                f"Offline dataset directory {directory} is incompatible with the first dataset manifest. "
                f"Mismatched fields: {mismatch_text}."
            )

    def _load_chunk(self, chunk_record: ChunkRecord) -> dict[str, np.ndarray]:
        with np.load(chunk_record.path, allow_pickle=False) as chunk_file:
            return {key: chunk_file[key] for key in chunk_file.files}

    def _prepare_chunk(self, chunk_record: ChunkRecord) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
        chunk = self._load_chunk(chunk_record)
        actions = chunk["action"].astype(np.int64, copy=False)
        done = chunk["done"].astype(np.float32, copy=False)
        timeout = chunk["timeout"].astype(np.float32, copy=False)
        episode_id = chunk["episode_id"].astype(np.int64, copy=False)
        player_index = chunk["player_index"].astype(np.int64, copy=False)

        valid_next = np.zeros(actions.shape[0], dtype=np.bool_)
        if actions.shape[0] > 1:
            valid_next[:-1] = (
                (episode_id[:-1] == episode_id[1:])
                & (player_index[:-1] == player_index[1:])
            )

        terminal_mask = (done == 1.0) & (timeout == 0.0)
        real_transition_mask = actions >= 0
        valid_transition_mask = real_transition_mask & (terminal_mask | valid_next)
        valid_indices = np.flatnonzero(valid_transition_mask)
        return chunk, valid_indices, terminal_mask

    def _activate_random_chunk(self) -> None:
        while True:
            chunk_index = int(np.random.choice(len(self.chunk_records), p=self.chunk_sampling_probs))
            chunk_record = self.chunk_records[chunk_index]
            chunk, valid_indices, terminal_mask = self._prepare_chunk(chunk_record)
            if valid_indices.size == 0:
                continue
            self._active_chunk_index = chunk_index
            self._active_chunk = chunk
            self._active_valid_indices = valid_indices
            self._active_terminal_mask = terminal_mask
            self._remaining_batches_on_active_chunk = self.chunk_reuse_batches
            return

    def sample_batch(
        self,
        batch_size: int,
        device: torch.device | str,
    ) -> dict[str, torch.Tensor]:
        device = torch.device(device)
        if (
            self._active_chunk is None
            or self._active_valid_indices is None
            or self._active_terminal_mask is None
            or self._remaining_batches_on_active_chunk <= 0
        ):
            self._activate_random_chunk()

        assert self._active_chunk is not None
        assert self._active_valid_indices is not None
        assert self._active_terminal_mask is not None

        chunk = self._active_chunk
        valid_indices = self._active_valid_indices
        terminal_mask = self._active_terminal_mask
        obs = chunk["obs"]
        actions = chunk["action"].astype(np.int64, copy=False)
        rewards = chunk[self.reward_key].astype(np.float32, copy=False)
        done = chunk["done"].astype(np.float32, copy=False)
        timeout = chunk["timeout"].astype(np.float32, copy=False)

        replace = valid_indices.size < batch_size
        sampled_indices = np.random.choice(valid_indices, size=batch_size, replace=replace)

        obs_batch = obs[sampled_indices]
        next_obs_batch = np.zeros_like(obs_batch)
        bootstrap_mask = ~terminal_mask[sampled_indices]
        if np.any(bootstrap_mask):
            next_obs_batch[bootstrap_mask] = obs[sampled_indices[bootstrap_mask] + 1]

        self._remaining_batches_on_active_chunk -= 1

        obs_tensor = torch.as_tensor(obs_batch, dtype=torch.float32, device=device)
        next_obs_tensor = torch.as_tensor(next_obs_batch, dtype=torch.float32, device=device)
        action_tensor = torch.as_tensor(actions[sampled_indices], dtype=torch.int64, device=device)
        reward_tensor = torch.as_tensor(rewards[sampled_indices], dtype=torch.float32, device=device)
        done_tensor = torch.as_tensor(done[sampled_indices], dtype=torch.float32, device=device)
        timeout_tensor = torch.as_tensor(timeout[sampled_indices], dtype=torch.float32, device=device)

        return {
            "obs": obs_tensor,
            "next_obs": next_obs_tensor,
            "action": action_tensor,
            "reward": reward_tensor,
            "done": done_tensor,
            "timeout": timeout_tensor,
        }

    def get_stats(self) -> dict[str, Any]:
        total_rows = 0
        total_transitions = 0
        unique_episodes: set[int] = set()
        action_counts = np.zeros(self.action_dim, dtype=np.int64)
        reward_sum = 0.0
        reward_sq_sum = 0.0
        episode_return_sums: dict[tuple[int, int], float] = {}
        reward_min = float("inf")
        reward_max = float("-inf")

        for chunk_record in self.chunk_records:
            chunk = self._load_chunk(chunk_record)
            actions = chunk["action"].astype(np.int64, copy=False)
            rewards = chunk[self.reward_key].astype(np.float32, copy=False)
            episode_ids = chunk["episode_id"].astype(np.int64, copy=False)
            player_indices = chunk["player_index"].astype(np.int64, copy=False)

            total_rows += int(actions.shape[0])
            real_mask = actions >= 0
            total_transitions += int(real_mask.sum())
            unique_episodes.update(int(value) for value in np.unique(episode_ids))

            if np.any(real_mask):
                valid_actions = actions[real_mask]
                action_counts += np.bincount(valid_actions, minlength=self.action_dim)
                valid_rewards = rewards[real_mask]
                reward_sum += float(valid_rewards.sum())
                reward_sq_sum += float(np.square(valid_rewards).sum())
                reward_min = min(reward_min, float(valid_rewards.min()))
                reward_max = max(reward_max, float(valid_rewards.max()))
                valid_episode_ids = episode_ids[real_mask]
                valid_player_indices = player_indices[real_mask]
                for episode_id, player_index, reward in zip(
                    valid_episode_ids,
                    valid_player_indices,
                    valid_rewards,
                    strict=True,
                ):
                    key = (int(episode_id), int(player_index))
                    episode_return_sums[key] = episode_return_sums.get(key, 0.0) + float(reward)

        reward_mean = reward_sum / max(total_transitions, 1)
        reward_var = reward_sq_sum / max(total_transitions, 1) - reward_mean**2
        reward_std = float(np.sqrt(max(reward_var, 0.0)))
        episode_returns = np.asarray(list(episode_return_sums.values()), dtype=np.float32)

        return {
            "num_chunks": len(self.chunk_records),
            "num_rows": total_rows,
            "num_transitions": total_transitions,
            "num_episodes": len(unique_episodes),
            "num_player_episodes": int(episode_returns.size),
            "action_distribution": action_counts.tolist(),
            "reward_mean": float(reward_mean),
            "reward_std": reward_std,
            "episode_return_mean": float(episode_returns.mean()) if episode_returns.size > 0 else float("nan"),
            "episode_return_std": float(episode_returns.std()) if episode_returns.size > 0 else float("nan"),
            "episode_return_min": float(episode_returns.min()) if episode_returns.size > 0 else float("nan"),
            "episode_return_max": float(episode_returns.max()) if episode_returns.size > 0 else float("nan"),
            "reward_min": float(reward_min) if total_transitions > 0 else float("nan"),
            "reward_max": float(reward_max) if total_transitions > 0 else float("nan"),
            "chunk_reuse_batches": self.chunk_reuse_batches,
        }
