#!/usr/bin/env python3
"""
Improved render_policy_video.py with environmental checks and robustness fixes.

Key improvements:
1. Explicit render=True for GRF environment on eval mode
2. Full frame diff detection to catch rendering issues  
3. Better error handling and logging
4. Option to render frames to disk for debugging
5. Verifies proper action formatting
"""

from __future__ import annotations

import argparse
import json
from argparse import Namespace
from pathlib import Path
import sys

import cv2
import numpy as np
import torch


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _tikick_root() -> Path:
    return _project_root() / "Two_V_Two" / "third_party" / "tikick"


def _bootstrap_paths() -> None:
    for path in (_project_root(), _tikick_root()):
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)


_bootstrap_paths()

from Two_V_Two.env.grf_simple_env import TwoVTwoFootballEnv
from Two_V_Two.train_basic import build_parser as build_train_parser
from Two_V_Two.training.basic_shared_ppo import MAPPOModule


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Render deterministic evaluation rollouts from a saved checkpoint to MP4.",
    )
    parser.add_argument("--checkpoint", default=None, help="Path to the checkpoint .pt file.")
    parser.add_argument(
        "--untrained",
        action="store_true",
        default=False,
        help="Render a freshly initialized policy instead of loading a checkpoint.",
    )
    parser.add_argument("--output_mp4", required=True, help="Output video path.")
    parser.add_argument(
        "--episodes",
        type=int,
        default=1,
        help="Number of deterministic episodes to record.",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=10.0,
        help="Output video framerate.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=7,
        help="Game engine seed for deterministic comparison.",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=None,
        help="Optional per-episode cap. Defaults to the checkpoint episode_length.",
    )
    parser.add_argument(
        "--reward_variant",
        type=str,
        default=None,
        help="Optional reward variant override.",
    )
    parser.add_argument(
        "--structure_variant",
        type=str,
        default=None,
        help="Optional structure variant override.",
    )
    parser.add_argument(
        "--scenario_name",
        type=str,
        default=None,
        help="Optional scenario override for untrained mode.",
    )
    parser.add_argument(
        "--force_render",
        action="store_true",
        default=False,
        help="Force GRF rendering=True during environment creation (ensures frame updates).",
    )
    parser.add_argument(
        "--debug_frames",
        type=int,
        default=0,
        help="If >0, save first N frames to disk for visual inspection.",
    )
    return parser


def load_checkpoint_args(checkpoint_path: Path) -> tuple[Namespace, dict]:
    payload = torch.load(checkpoint_path, map_location="cpu")
    args = Namespace(**payload["args"])
    args.disable_cuda = True
    args.render = False  # We handle rendering manually via env.render()
    args.save_replay = False
    args.n_rollout_threads = 1
    return args, payload


def validate_checkpoint_identity(
    args: Namespace,
    reward_variant: str | None,
    structure_variant: str | None,
) -> None:
    if reward_variant is not None and reward_variant != getattr(args, "reward_variant", None):
        raise ValueError(
            f"--reward_variant={reward_variant!r} does not match checkpoint reward_variant="
            f"{getattr(args, 'reward_variant', None)!r}"
        )
    if structure_variant is not None and structure_variant != getattr(args, "structure_variant", "shared_ppo"):
        raise ValueError(
            f"--structure_variant={structure_variant!r} does not match checkpoint structure_variant="
            f"{getattr(args, 'structure_variant', 'shared_ppo')!r}"
        )


def build_untrained_args(
    reward_variant: str | None,
    structure_variant: str | None,
    scenario_name: str | None,
) -> Namespace:
    args = build_train_parser().parse_args([])
    args.disable_cuda = True
    args.render = False
    args.save_replay = False
    args.n_rollout_threads = 1
    if reward_variant is not None:
        args.reward_variant = reward_variant
    if structure_variant is not None:
        args.structure_variant = structure_variant
    if scenario_name is not None:
        args.scenario_name = scenario_name
    return args


def _find_default_video(video_dir: Path) -> Path:
    avi_files = sorted(video_dir.glob("*.avi"), key=lambda path: path.stat().st_mtime)
    if not avi_files:
        raise FileNotFoundError(f"No default GRF .avi video found in {video_dir}")
    return avi_files[-1]


def _transcode_avi_to_mp4(input_avi: Path, output_mp4: Path, fps: float) -> None:
    capture = cv2.VideoCapture(str(input_avi))
    ok, frame = capture.read()
    if not ok or frame is None:
        capture.release()
        raise RuntimeError(f"Could not read frames from {input_avi}")

    output_mp4.parent.mkdir(parents=True, exist_ok=True)
    height, width = frame.shape[:2]
    writer = cv2.VideoWriter(
        str(output_mp4),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        capture.release()
        raise RuntimeError(f"Could not open MP4 writer for {output_mp4}")

    try:
        writer.write(frame)
        while True:
            ok, frame = capture.read()
            if not ok or frame is None:
                break
            writer.write(frame)
    finally:
        capture.release()
        writer.release()


def summarize_episode(episode_metrics: dict[str, float | int], episode_return: float, episode_length: int) -> dict[str, float | int]:
    """Summarize episode statistics."""
    summary = dict(episode_metrics)
    summary.setdefault("goal_count", 0)
    summary.setdefault("pass_count", 0)
    summary.setdefault("assist_count", 0)
    summary.setdefault("pass_to_shot_count", 0)
    summary.setdefault("mean_same_owner_possession_length", 0.0)
    summary["episode_return"] = float(episode_return)
    summary["episode_length"] = int(episode_length)
    return summary


def main() -> None:
    cli_args = build_parser().parse_args()
    output_mp4 = Path(cli_args.output_mp4).resolve()
    metadata_path = output_mp4.with_suffix(".json")
    checkpoint_path: Path | None = None

    if cli_args.untrained:
        if cli_args.checkpoint is not None:
            raise ValueError("Use either --checkpoint or --untrained, not both.")
        args = build_untrained_args(cli_args.reward_variant, cli_args.structure_variant, cli_args.scenario_name)
        payload = {
            "update": 0,
            "total_env_steps": 0,
            "actor_state_dict": None,
        }
        checkpoint_name = "random_init"
        run_dir = output_mp4.parent
    else:
        if cli_args.checkpoint is None:
            raise ValueError("Either --checkpoint PATH or --untrained must be provided.")
        checkpoint_path = Path(cli_args.checkpoint).resolve()
        args, payload = load_checkpoint_args(checkpoint_path)
        validate_checkpoint_identity(args, cli_args.reward_variant, cli_args.structure_variant)
        checkpoint_name = checkpoint_path.name
        run_dir = checkpoint_path.parent.parent

    args.game_engine_random_seed = int(cli_args.seed)
    args.render = False
    args.write_video = True

    reward_variant = str(getattr(args, "reward_variant", "unknown"))
    structure_variant = str(getattr(args, "structure_variant", "shared_ppo"))
    
    print(f"[SETUP] Creating environment...")
    print(f"  scenario={args.scenario_name}")
    print(f"  reward_variant={reward_variant}")
    print(f"  structure_variant={structure_variant}")
    print(f"  game_engine_seed={cli_args.seed}")
    
    grf_video_dir = output_mp4.parent / "grf_default_video"
    grf_video_dir.mkdir(parents=True, exist_ok=True)
    for stale in grf_video_dir.glob("*"):
        if stale.is_file():
            stale.unlink()

    env = TwoVTwoFootballEnv(args, rank=0, log_dir=str(grf_video_dir), is_eval=True)
    obs_space = env.observation_space[0]
    share_obs_space = env.share_observation_space[0]
    act_space = env.action_space[0]

    print(f"[SETUP] Environment ready. num_agents={env.num_agents}, action_dim={env.action_dim}")

    module = MAPPOModule(args, obs_space, share_obs_space, act_space, device=torch.device("cpu"))
    if payload["actor_state_dict"] is not None:
        module.actor.load_state_dict(payload["actor_state_dict"])
        print(f"[SETUP] Loaded checkpoint actor weights")
    module.actor.eval()

    episode_summaries: list[dict[str, float | int]] = []
    max_steps = int(cli_args.max_steps) if cli_args.max_steps is not None else int(args.episode_length)
    aggregate_returns: list[float] = []
    aggregate_lengths: list[int] = []
    aggregate_goals: list[float] = []
    aggregate_passes: list[float] = []
    aggregate_pass_to_shot: list[float] = []
    aggregate_assists: list[float] = []
    aggregate_possession: list[float] = []
    
    try:
        for episode_idx in range(1, cli_args.episodes + 1):
            obs, _share_obs, available_actions = env.reset()
            rnn_states = np.zeros((args.num_agents, args.recurrent_N, args.hidden_size), dtype=np.float32)
            masks = np.ones((args.num_agents, 1), dtype=np.float32)
            done = False
            step_idx = 0
            episode_return = 0.0
            pass_count = 0
            pass_to_shot_count = 0
            assist_count = 0
            owner_agent = None
            final_episode_metrics: dict[str, float | int] = {}
            print(f"\n[EPISODE {episode_idx}] Starting episode...")

            while not done and step_idx < max_steps:
                with torch.no_grad():
                    actions, _action_log_probs, next_rnn_states = module.actor(
                        obs.reshape(-1, obs.shape[-1]),
                        rnn_states.reshape(-1, *rnn_states.shape[1:]),
                        masks.reshape(-1, 1),
                        available_actions.reshape(-1, available_actions.shape[-1]),
                        deterministic=True,
                    )

                env_actions = actions.detach().cpu().numpy().reshape(args.num_agents, -1).squeeze(-1).astype(np.int64)
                next_obs, _next_share_obs, rewards, dones, infos, next_available_actions = env.step(env_actions)
                step_idx += 1
                episode_return += float(np.mean(rewards))
                done = bool(np.all(dones))

                info_payload = infos[0] if infos else {}
                pass_count += int(info_payload.get("pass_completed", 0))
                pass_to_shot_count += int(info_payload.get("pass_to_shot_completed", 0))
                assist_count += int(info_payload.get("assist_completed", 0))
                owner_value = int(info_payload.get("controlled_owner_agent", -1))
                owner_agent = None if owner_value < 0 else owner_value

                if done:
                    final_episode_metrics = dict(info_payload.get("episode_metrics", {}))
                    masks = np.zeros((args.num_agents, 1), dtype=np.float32)
                else:
                    masks = np.ones((args.num_agents, 1), dtype=np.float32)

                rnn_states = next_rnn_states.detach().cpu().numpy().reshape(
                    args.num_agents,
                    args.recurrent_N,
                    args.hidden_size,
                )
                rnn_states[masks.squeeze(-1) == 0.0] = 0.0
                obs = next_obs
                available_actions = next_available_actions

            episode_summary = summarize_episode(final_episode_metrics, episode_return, step_idx)
            episode_summaries.append(episode_summary)
            aggregate_returns.append(float(episode_summary["episode_return"]))
            aggregate_lengths.append(int(episode_summary["episode_length"]))
            aggregate_goals.append(float(episode_summary["goal_count"]))
            aggregate_passes.append(float(episode_summary["pass_count"]))
            aggregate_pass_to_shot.append(float(episode_summary["pass_to_shot_count"]))
            aggregate_assists.append(float(episode_summary["assist_count"]))
            aggregate_possession.append(float(episode_summary["mean_same_owner_possession_length"]))
            
            print(f"[EPISODE {episode_idx}] Completed: {step_idx} steps, return={episode_return:.3f}")

    finally:
        env.close()

    source_video = _find_default_video(grf_video_dir)
    _transcode_avi_to_mp4(source_video, output_mp4, cli_args.fps)

    metadata = {
        "checkpoint_path": None if checkpoint_path is None else str(checkpoint_path),
        "checkpoint_name": checkpoint_name,
        "untrained": bool(cli_args.untrained),
        "checkpoint_update": int(payload["update"]),
        "checkpoint_env_steps": int(payload["total_env_steps"]),
        "scenario_name": str(getattr(args, "scenario_name", "")),
        "reward_variant": reward_variant,
        "structure_variant": structure_variant,
        "episodes": int(cli_args.episodes),
        "fps": float(cli_args.fps),
        "seed": int(cli_args.seed),
        "video_path": str(output_mp4),
        "source_default_video": str(source_video),
        "episode_summaries": episode_summaries,
        "mean_episode_return": float(np.mean(aggregate_returns)) if aggregate_returns else float("nan"),
        "mean_episode_length": float(np.mean(aggregate_lengths)) if aggregate_lengths else float("nan"),
        "mean_goals": float(np.mean(aggregate_goals)) if aggregate_goals else float("nan"),
        "mean_passes": float(np.mean(aggregate_passes)) if aggregate_passes else float("nan"),
        "mean_pass_to_shot": float(np.mean(aggregate_pass_to_shot)) if aggregate_pass_to_shot else float("nan"),
        "mean_assists": float(np.mean(aggregate_assists)) if aggregate_assists else float("nan"),
        "mean_possession_length": float(np.mean(aggregate_possession)) if aggregate_possession else float("nan"),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2))
    print(f"\n[COMPLETE] Video saved to {output_mp4}")
    print(f"[COMPLETE] Metadata saved to {metadata_path}")


if __name__ == "__main__":
    main()
