from __future__ import annotations

import argparse
import json
import math
import time
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.distributions import Categorical

from yfu_football.envs import FootballVecEnv, RewardShapingConfig
from yfu_football.iql import DiscreteIQL
from yfu_football.model import ActorCritic
from yfu_football.ppo import _set_seed


MAX_MATCH_STEPS = 3001


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect offline GRF data from the Y_Fu PPO policy or a random policy.")
    parser.add_argument("--checkpoint")
    parser.add_argument("--policy", choices=("checkpoint", "random"), default="checkpoint")
    parser.add_argument("--num-envs", type=int, default=6)
    parser.add_argument("--total-env-steps", type=int, required=True)
    parser.add_argument("--epsilon", type=float, default=0.0)
    parser.add_argument("--chunk-size", type=int, default=500_000)
    parser.add_argument("--save-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--obs-dtype", choices=("float16", "uint8"), default="float16")
    parser.add_argument("--checkpoint-id", type=int, default=0)
    return parser.parse_args()


def _resolve_device(_: str) -> torch.device:
    return torch.device("cpu")


def _reward_shaping_from_config(config: dict[str, Any]) -> RewardShapingConfig:
    return RewardShapingConfig(
        pass_success_reward=float(config.get("pass_success_reward", 0.0)),
        pass_failure_penalty=float(config.get("pass_failure_penalty", 0.0)),
        pass_progress_reward_scale=float(config.get("pass_progress_reward_scale", 0.0)),
        shot_attempt_reward=float(config.get("shot_attempt_reward", 0.0)),
        attacking_possession_reward=float(config.get("attacking_possession_reward", 0.0)),
        attacking_x_threshold=float(config.get("attacking_x_threshold", 0.55)),
        final_third_entry_reward=float(config.get("final_third_entry_reward", 0.0)),
        possession_retention_reward=float(config.get("possession_retention_reward", 0.0)),
        possession_recovery_reward=float(config.get("possession_recovery_reward", 0.0)),
        defensive_third_recovery_reward=float(config.get("defensive_third_recovery_reward", 0.0)),
        opponent_attacking_possession_penalty=float(config.get("opponent_attacking_possession_penalty", 0.0)),
        own_half_turnover_penalty=float(config.get("own_half_turnover_penalty", 0.0)),
        own_half_x_threshold=float(config.get("own_half_x_threshold", 0.0)),
        defensive_x_threshold=float(config.get("defensive_x_threshold", -0.45)),
        pending_pass_horizon=int(config.get("pending_pass_horizon", 8)),
    )


def _zero_reward_shaping() -> RewardShapingConfig:
    return RewardShapingConfig()


def load_checkpoint_policy(
    checkpoint_path: str,
    device: torch.device,
) -> tuple[dict[str, Any], ActorCritic | DiscreteIQL]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    config = checkpoint.get("config", {})
    if "model_state_dict" in checkpoint:
        model = ActorCritic(
            obs_dim=checkpoint["obs_dim"],
            action_dim=checkpoint["action_dim"],
            hidden_sizes=tuple(config.get("hidden_sizes", [256, 256])),
            obs_shape=tuple(checkpoint.get("obs_shape", (checkpoint["obs_dim"],))),
            model_type=config.get("model_type", "auto"),
            feature_dim=int(config.get("feature_dim", 256)),
            player_id_dim=int(config.get("num_players", config.get("num_controlled_players", 0)))
            if bool(config.get("use_player_id", False))
            else 0,
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)
        model.eval()
        return config, model
    if "q1_state_dict" in checkpoint and "q2_state_dict" in checkpoint and "v_state_dict" in checkpoint:
        model = DiscreteIQL.load_checkpoint(checkpoint_path, device=device)
        return config, model
    raise ValueError(
        f"Unsupported checkpoint format for offline collection: {checkpoint_path}. "
        "Expected a PPO checkpoint with 'model_state_dict' or an IQL checkpoint "
        "with 'q1_state_dict'/'q2_state_dict'/'v_state_dict'."
    )


def _obs_dtype_numpy(obs_dtype: str) -> np.dtype[Any]:
    if obs_dtype == "float16":
        return np.float16
    return np.uint8


def _convert_observation(observation: np.ndarray, obs_dtype: str) -> np.ndarray:
    if obs_dtype == "float16":
        return np.asarray(observation, dtype=np.float16)
    clipped = np.clip(np.rint(observation), 0, 255)
    return clipped.astype(np.uint8)


class ChunkWriter:
    def __init__(
        self,
        *,
        save_dir: Path,
        chunk_size: int,
        obs_dim: int,
        obs_dtype: str,
    ) -> None:
        self.save_dir = save_dir
        self.chunk_size = int(chunk_size)
        self.obs_dim = int(obs_dim)
        self.obs_dtype = obs_dtype
        self._chunk_index = 0
        self._count = 0
        self.chunk_metadata: list[dict[str, Any]] = []
        numpy_obs_dtype = _obs_dtype_numpy(obs_dtype)
        self.obs = np.zeros((self.chunk_size, self.obs_dim), dtype=numpy_obs_dtype)
        self.action = np.zeros(self.chunk_size, dtype=np.int8)
        self.reward = np.zeros(self.chunk_size, dtype=np.float16)
        self.score_reward = np.zeros(self.chunk_size, dtype=np.float16)
        self.done = np.zeros(self.chunk_size, dtype=np.uint8)
        self.timeout = np.zeros(self.chunk_size, dtype=np.uint8)
        self.log_prob = np.zeros(self.chunk_size, dtype=np.float16)
        self.player_index = np.zeros(self.chunk_size, dtype=np.uint8)
        self.episode_id = np.zeros(self.chunk_size, dtype=np.int32)
        self.checkpoint_id = np.zeros(self.chunk_size, dtype=np.uint8)

    @property
    def count(self) -> int:
        return self._count

    def add_sequence(self, sequence: dict[str, np.ndarray]) -> None:
        seq_len = int(sequence["action"].shape[0])
        if seq_len == 0:
            return
        if self._count > 0 and self._count + seq_len > self.chunk_size:
            self.flush()
        if seq_len > self.chunk_size:
            raise ValueError(f"Single trajectory of length {seq_len} exceeds chunk size {self.chunk_size}.")

        start = self._count
        stop = self._count + seq_len
        self.obs[start:stop] = sequence["obs"]
        self.action[start:stop] = sequence["action"]
        self.reward[start:stop] = sequence["reward"]
        self.score_reward[start:stop] = sequence["score_reward"]
        self.done[start:stop] = sequence["done"]
        self.timeout[start:stop] = sequence["timeout"]
        self.log_prob[start:stop] = sequence["log_prob"]
        self.player_index[start:stop] = sequence["player_index"]
        self.episode_id[start:stop] = sequence["episode_id"]
        self.checkpoint_id[start:stop] = sequence["checkpoint_id"]
        self._count = stop

    def flush(self) -> None:
        if self._count == 0:
            return
        filename = f"chunk_{self._chunk_index:05d}.npz"
        path = self.save_dir / filename
        np.savez_compressed(
            path,
            obs=self.obs[: self._count],
            action=self.action[: self._count],
            reward=self.reward[: self._count],
            score_reward=self.score_reward[: self._count],
            done=self.done[: self._count],
            timeout=self.timeout[: self._count],
            log_prob=self.log_prob[: self._count],
            player_index=self.player_index[: self._count],
            episode_id=self.episode_id[: self._count],
            checkpoint_id=self.checkpoint_id[: self._count],
        )
        real_transitions = int(np.sum(self.action[: self._count] >= 0))
        self.chunk_metadata.append(
            {
                "filename": filename,
                "num_rows": int(self._count),
                "num_transitions": real_transitions,
            }
        )
        self._count = 0
        self._chunk_index += 1


def _new_sequence() -> dict[str, list[Any]]:
    return {
        "obs": [],
        "action": [],
        "reward": [],
        "score_reward": [],
        "done": [],
        "timeout": [],
        "log_prob": [],
        "player_index": [],
        "episode_id": [],
        "checkpoint_id": [],
    }


def _append_transition(
    sequence: dict[str, list[Any]],
    *,
    obs: np.ndarray,
    action: int,
    reward: float,
    score_reward: float,
    done: int,
    timeout: int,
    log_prob: float,
    player_index: int,
    episode_id: int,
    checkpoint_id: int,
) -> None:
    sequence["obs"].append(obs)
    sequence["action"].append(int(action))
    sequence["reward"].append(float(reward))
    sequence["score_reward"].append(float(score_reward))
    sequence["done"].append(int(done))
    sequence["timeout"].append(int(timeout))
    sequence["log_prob"].append(float(log_prob))
    sequence["player_index"].append(int(player_index))
    sequence["episode_id"].append(int(episode_id))
    sequence["checkpoint_id"].append(int(checkpoint_id))


def _append_timeout_bootstrap_row(
    sequence: dict[str, list[Any]],
    *,
    terminal_observation: np.ndarray,
    player_index: int,
    episode_id: int,
    checkpoint_id: int,
) -> None:
    _append_transition(
        sequence,
        obs=terminal_observation,
        action=-1,
        reward=0.0,
        score_reward=0.0,
        done=1,
        timeout=1,
        log_prob=0.0,
        player_index=player_index,
        episode_id=episode_id,
        checkpoint_id=checkpoint_id,
    )


def _finalize_sequence(
    sequence: dict[str, list[Any]],
    *,
    obs_dtype: str,
) -> dict[str, np.ndarray]:
    return {
        "obs": np.stack(sequence["obs"], axis=0).astype(_obs_dtype_numpy(obs_dtype), copy=False),
        "action": np.asarray(sequence["action"], dtype=np.int8),
        "reward": np.asarray(sequence["reward"], dtype=np.float16),
        "score_reward": np.asarray(sequence["score_reward"], dtype=np.float16),
        "done": np.asarray(sequence["done"], dtype=np.uint8),
        "timeout": np.asarray(sequence["timeout"], dtype=np.uint8),
        "log_prob": np.asarray(sequence["log_prob"], dtype=np.float16),
        "player_index": np.asarray(sequence["player_index"], dtype=np.uint8),
        "episode_id": np.asarray(sequence["episode_id"], dtype=np.int32),
        "checkpoint_id": np.asarray(sequence["checkpoint_id"], dtype=np.uint8),
    }


def _sample_actions_from_model(
    model: ActorCritic | DiscreteIQL,
    observation: np.ndarray,
    *,
    epsilon: float,
    num_actions: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    num_envs, num_players, obs_dim = observation.shape
    obs_tensor = torch.as_tensor(
        observation.reshape(num_envs * num_players, obs_dim),
        dtype=torch.float32,
        device=device,
    )
    with torch.no_grad():
        if isinstance(model, ActorCritic):
            player_ids = None
            if getattr(model, "player_id_dim", 0) > 0:
                player_ids = torch.arange(num_players, dtype=torch.int64, device=device).repeat(num_envs)
            logits, _ = model.forward(obs_tensor, player_ids=player_ids)
        else:
            logits = model.policy_logits(obs_tensor) / max(float(model.config.temperature), 1e-6)
        distribution = Categorical(logits=logits)
        sampled_actions = distribution.sample()
        probs = torch.softmax(logits, dim=-1)
    sampled_actions_np = sampled_actions.cpu().numpy().reshape(num_envs, num_players)
    probs_np = probs.cpu().numpy().reshape(num_envs, num_players, num_actions)

    epsilon_mask = np.random.random(size=sampled_actions_np.shape) < float(epsilon)
    random_actions = np.random.randint(num_actions, size=sampled_actions_np.shape, dtype=np.int64)
    final_actions = np.where(epsilon_mask, random_actions, sampled_actions_np)

    selected_probs = probs_np[
        np.arange(num_envs)[:, None],
        np.arange(num_players)[None, :],
        final_actions,
    ]
    behavior_probs = (1.0 - float(epsilon)) * selected_probs + float(epsilon) / float(num_actions)
    behavior_probs = np.clip(behavior_probs, 1e-8, 1.0)
    log_probs = np.log(behavior_probs).astype(np.float32)
    return final_actions.astype(np.int64), log_probs


def _sample_random_actions(
    observation: np.ndarray,
    *,
    num_actions: int,
) -> tuple[np.ndarray, np.ndarray]:
    num_envs, num_players, _ = observation.shape
    actions = np.random.randint(num_actions, size=(num_envs, num_players), dtype=np.int64)
    log_prob = np.full((num_envs, num_players), -math.log(float(num_actions)), dtype=np.float32)
    return actions, log_prob


def main() -> None:
    args = parse_args()
    if args.policy == "checkpoint" and not args.checkpoint:
        raise ValueError("--checkpoint is required when --policy checkpoint is used.")

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    _set_seed(args.seed)

    device = _resolve_device("cpu")
    checkpoint_config: dict[str, Any] = {}
    model: ActorCritic | DiscreteIQL | None = None
    if args.policy == "checkpoint":
        checkpoint_config, model = load_checkpoint_policy(args.checkpoint, device)

    env_name = checkpoint_config.get("env_name", "5_vs_5")
    representation = checkpoint_config.get("representation", "extracted")
    rewards = checkpoint_config.get("rewards", "scoring,checkpoints")
    num_controlled_players = int(checkpoint_config.get("num_controlled_players", 4))
    channel_dimensions = tuple(checkpoint_config.get("channel_dimensions", (42, 42)))

    env = FootballVecEnv(
        num_envs=args.num_envs,
        base_seed=args.seed,
        env_name=env_name,
        representation=representation,
        rewards=rewards,
        render=False,
        logdir="",
        num_controlled_players=num_controlled_players,
        channel_dimensions=channel_dimensions,
        reward_shaping=_zero_reward_shaping(),
    )

    writer = ChunkWriter(
        save_dir=save_dir,
        chunk_size=args.chunk_size,
        obs_dim=env.obs_dim,
        obs_dtype=args.obs_dtype,
    )

    observation = env.reset()
    num_actions = env.action_dim
    episode_ids = [env_index for env_index in range(args.num_envs)]
    next_episode_id = args.num_envs
    episode_lengths = np.zeros(args.num_envs, dtype=np.int32)
    trajectory_buffers = [
        [_new_sequence() for _ in range(env.num_players)]
        for _ in range(args.num_envs)
    ]

    total_env_steps = 0
    total_real_transitions = 0
    total_rows = 0
    completed_episodes = 0
    last_scores = deque(maxlen=100)
    collect_start = time.perf_counter()
    last_log_time = collect_start

    try:
        while total_env_steps < args.total_env_steps:
            step_start = time.perf_counter()
            if args.policy == "random":
                actions, log_probs = _sample_random_actions(observation, num_actions=num_actions)
            else:
                assert model is not None
                actions, log_probs = _sample_actions_from_model(
                    model,
                    observation,
                    epsilon=args.epsilon,
                    num_actions=num_actions,
                    device=device,
                )

            next_observation, reward, done, infos = env.step(actions)
            episode_lengths += 1

            for env_index in range(args.num_envs):
                episode_done = bool(done[env_index])
                timeout = int(episode_done and episode_lengths[env_index] >= MAX_MATCH_STEPS)
                info = infos[env_index]
                score_reward = float(info.get("score_reward", 0.0))
                for player_index in range(env.num_players):
                    _append_transition(
                        trajectory_buffers[env_index][player_index],
                        obs=_convert_observation(observation[env_index, player_index], args.obs_dtype),
                        action=int(actions[env_index, player_index]),
                        reward=float(reward[env_index, player_index]),
                        score_reward=score_reward,
                        done=int(episode_done),
                        timeout=timeout,
                        log_prob=float(log_probs[env_index, player_index]),
                        player_index=player_index,
                        episode_id=episode_ids[env_index],
                        checkpoint_id=args.checkpoint_id,
                    )
                    total_real_transitions += 1

                if episode_done:
                    completed_episodes += 1
                    final_score = info.get("final_score")
                    if final_score is not None:
                        last_scores.append((int(final_score[0]), int(final_score[1])))

                    terminal_observation = np.asarray(
                        info.get("terminal_observation", observation[env_index]),
                        dtype=np.float32,
                    ).reshape(env.num_players, env.obs_dim)

                    for player_index in range(env.num_players):
                        if timeout:
                            _append_timeout_bootstrap_row(
                                trajectory_buffers[env_index][player_index],
                                terminal_observation=_convert_observation(
                                    terminal_observation[player_index],
                                    args.obs_dtype,
                                ),
                                player_index=player_index,
                                episode_id=episode_ids[env_index],
                                checkpoint_id=args.checkpoint_id,
                            )
                            total_rows += 1

                        finalized = _finalize_sequence(
                            trajectory_buffers[env_index][player_index],
                            obs_dtype=args.obs_dtype,
                        )
                        writer.add_sequence(finalized)
                        total_rows += int(finalized["action"].shape[0]) - int(timeout)
                        trajectory_buffers[env_index][player_index] = _new_sequence()

                    episode_ids[env_index] = next_episode_id
                    next_episode_id += 1
                    episode_lengths[env_index] = 0

            total_env_steps += args.num_envs
            observation = next_observation

            now = time.perf_counter()
            if now - last_log_time >= 10.0:
                elapsed = max(now - collect_start, 1e-8)
                recent_env_fps = args.num_envs / max(now - step_start, 1e-8)
                mean_goals_for = float(np.mean([score[0] for score in last_scores])) if last_scores else float("nan")
                mean_goals_against = float(np.mean([score[1] for score in last_scores])) if last_scores else float("nan")
                win_rate = (
                    float(np.mean([1.0 if left > right else 0.0 for left, right in last_scores]))
                    if last_scores
                    else float("nan")
                )
                print(
                    f"[collect] env_steps={total_env_steps} "
                    f"transitions={total_real_transitions} "
                    f"episodes={completed_episodes} "
                    f"env_fps={recent_env_fps:.1f} "
                    f"avg_env_fps={total_env_steps / elapsed:.1f} "
                    f"goals_for={mean_goals_for:.2f} "
                    f"goals_against={mean_goals_against:.2f} "
                    f"win_rate={win_rate:.3f}"
                )
                last_log_time = now
    finally:
        for env_index in range(args.num_envs):
            for player_index in range(env.num_players):
                if trajectory_buffers[env_index][player_index]["action"]:
                    finalized = _finalize_sequence(
                        trajectory_buffers[env_index][player_index],
                        obs_dtype=args.obs_dtype,
                    )
                    writer.add_sequence(finalized)
                    total_rows += int(finalized["action"].shape[0])
        writer.flush()
        env.close()

    total_rows_written = int(sum(chunk["num_rows"] for chunk in writer.chunk_metadata))
    total_transitions_written = int(sum(chunk["num_transitions"] for chunk in writer.chunk_metadata))
    manifest = {
        "policy": args.policy,
        "checkpoint": args.checkpoint,
        "checkpoint_id": args.checkpoint_id,
        "seed": args.seed,
        "epsilon": args.epsilon,
        "num_envs": args.num_envs,
        "total_env_steps": total_env_steps,
        "total_player_transitions": total_transitions_written,
        "total_rows": total_rows_written,
        "episodes_completed": completed_episodes,
        "obs_dtype": args.obs_dtype,
        "obs_dim": env.obs_dim,
        "obs_shape": list(env.obs_shape),
        "action_dim": env.action_dim,
        "env_name": env_name,
        "representation": representation,
        "rewards": rewards,
        "num_controlled_players": num_controlled_players,
        "channel_dimensions": list(channel_dimensions),
        "reward_shaping": {
            "pass_success_reward": 0.0,
            "pass_failure_penalty": 0.0,
            "pass_progress_reward_scale": 0.0,
            "shot_attempt_reward": 0.0,
            "attacking_possession_reward": 0.0,
            "attacking_x_threshold": 0.55,
            "final_third_entry_reward": 0.0,
            "possession_retention_reward": 0.0,
            "possession_recovery_reward": 0.0,
            "defensive_third_recovery_reward": 0.0,
            "opponent_attacking_possession_penalty": 0.0,
            "own_half_turnover_penalty": 0.0,
            "own_half_x_threshold": 0.0,
            "defensive_x_threshold": -0.45,
            "pending_pass_horizon": 8,
        },
        "source_checkpoint_reward_shaping": {
            "pass_success_reward": float(checkpoint_config.get("pass_success_reward", 0.0)),
            "pass_failure_penalty": float(checkpoint_config.get("pass_failure_penalty", 0.0)),
            "pass_progress_reward_scale": float(checkpoint_config.get("pass_progress_reward_scale", 0.0)),
            "shot_attempt_reward": float(checkpoint_config.get("shot_attempt_reward", 0.0)),
            "attacking_possession_reward": float(checkpoint_config.get("attacking_possession_reward", 0.0)),
            "attacking_x_threshold": float(checkpoint_config.get("attacking_x_threshold", 0.55)),
            "final_third_entry_reward": float(checkpoint_config.get("final_third_entry_reward", 0.0)),
            "possession_retention_reward": float(checkpoint_config.get("possession_retention_reward", 0.0)),
            "possession_recovery_reward": float(checkpoint_config.get("possession_recovery_reward", 0.0)),
            "defensive_third_recovery_reward": float(checkpoint_config.get("defensive_third_recovery_reward", 0.0)),
            "opponent_attacking_possession_penalty": float(checkpoint_config.get("opponent_attacking_possession_penalty", 0.0)),
            "own_half_turnover_penalty": float(checkpoint_config.get("own_half_turnover_penalty", 0.0)),
            "own_half_x_threshold": float(checkpoint_config.get("own_half_x_threshold", 0.0)),
            "defensive_x_threshold": float(checkpoint_config.get("defensive_x_threshold", -0.45)),
            "pending_pass_horizon": int(checkpoint_config.get("pending_pass_horizon", 8)),
        },
        "chunks": writer.chunk_metadata,
    }
    (save_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Collection finished. Manifest: {save_dir / 'manifest.json'}")


if __name__ == "__main__":
    main()
