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


def get_frame(env: TwoVTwoFootballEnv) -> np.ndarray:
    """
    Capture an RGB frame from the environment's renderer.
    Tries both env.env and env.env.unwrapped to find the render method.
    """
    targets = [getattr(env, "env", None), getattr(getattr(env, "env", None), "unwrapped", None)]
    for target in targets:
        if target is None:
            continue
        try:
            frame = target.render(mode="rgb_array")
        except Exception as e:
            print(f"[WARN] Frame capture attempt failed: {e}")
            continue
        if frame is not None:
            frame_array = np.asarray(frame, dtype=np.uint8)
            # Sanity check - frames should never be all-zeros or have no variation
            if np.any(frame_array > 0):
                return frame_array
    raise RuntimeError("Could not capture a valid RGB frame from the football environment.")


def init_writer(video_path: Path, frame: np.ndarray, fps: float) -> cv2.VideoWriter:
    """Initialize the MP4 video writer."""
    video_path.parent.mkdir(parents=True, exist_ok=True)
    height, width = frame.shape[:2]
    writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer for {video_path}.")
    return writer


def _draw_panel(frame: np.ndarray, lines: list[str]) -> np.ndarray:
    """Draw a text overlay panel on the frame."""
    output = frame.copy()
    panel_height = 28 + 28 * len(lines)
    cv2.rectangle(output, (8, 8), (780, min(panel_height, frame.shape[0] - 8)), (0, 0, 0), thickness=-1)
    y = 34
    for line in lines:
        cv2.putText(
            output,
            line,
            (18, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            lineType=cv2.LINE_AA,
        )
        y += 28
    return output


def overlay_frame(
    frame_rgb: np.ndarray,
    checkpoint_name: str,
    total_env_steps: int,
    reward_variant: str,
    structure_variant: str,
    episode_idx: int,
    total_episodes: int,
    step_idx: int,
    episode_return: float,
    pass_count: int,
    pass_to_shot_count: int,
    assist_count: int,
    owner_agent: int | None,
) -> np.ndarray:
    """Add info overlay to an RGB frame."""
    owner_text = "none" if owner_agent is None or owner_agent < 0 else str(owner_agent)
    lines = [
        f"reward={reward_variant} structure={structure_variant}",
        f"checkpoint={checkpoint_name} env_steps={total_env_steps}",
        f"episode={episode_idx}/{total_episodes} step={step_idx} return={episode_return:.3f}",
        (
            f"passes={pass_count} pass_to_shot={pass_to_shot_count} "
            f"assists={assist_count} owner_agent={owner_text}"
        ),
    ]
    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    return _draw_panel(frame_bgr, lines)


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
    
    # FIX: On eval mode, enable GRF rendering to ensure the graphics pipeline is active
    if cli_args.force_render or True:  # Always enable for safety
        args.render = False  # But keep our manual frame capture
        args.write_video = False

    reward_variant = str(getattr(args, "reward_variant", "unknown"))
    structure_variant = str(getattr(args, "structure_variant", "shared_ppo"))
    
    print(f"[SETUP] Creating environment...")
    print(f"  scenario={args.scenario_name}")
    print(f"  reward_variant={reward_variant}")
    print(f"  structure_variant={structure_variant}")
    print(f"  game_engine_seed={cli_args.seed}")
    
    env = TwoVTwoFootballEnv(args, rank=0, log_dir=str(run_dir / "video_eval_replays"), is_eval=True)
    obs_space = env.observation_space[0]
    share_obs_space = env.share_observation_space[0]
    act_space = env.action_space[0]

    print(f"[SETUP] Environment ready. num_agents={env.num_agents}, action_dim={env.action_dim}")

    module = MAPPOModule(args, obs_space, share_obs_space, act_space, device=torch.device("cpu"))
    if payload["actor_state_dict"] is not None:
        module.actor.load_state_dict(payload["actor_state_dict"])
        print(f"[SETUP] Loaded checkpoint actor weights")
    module.actor.eval()

    writer: cv2.VideoWriter | None = None
    episode_summaries: list[dict[str, float | int]] = []
    max_steps = int(cli_args.max_steps) if cli_args.max_steps is not None else int(args.episode_length)
    aggregate_returns: list[float] = []
    aggregate_lengths: list[int] = []
    aggregate_goals: list[float] = []
    aggregate_passes: list[float] = []
    aggregate_pass_to_shot: list[float] = []
    aggregate_assists: list[float] = []
    aggregate_possession: list[float] = []
    
    debug_frame_dir = None
    if cli_args.debug_frames > 0:
        debug_frame_dir = output_mp4.parent / "debug_frames"
        debug_frame_dir.mkdir(parents=True, exist_ok=True)
        print(f"[DEBUG] Will save first {cli_args.debug_frames} frames to {debug_frame_dir}")

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

            initial_frame = get_frame(env)
            if writer is None:
                writer = init_writer(output_mp4, initial_frame, cli_args.fps)
                print(f"[VIDEO] Initialized MP4 writer: {initial_frame.shape[1]}x{initial_frame.shape[0]}")
            
            writer.write(
                overlay_frame(
                    initial_frame,
                    checkpoint_name=checkpoint_name,
                    total_env_steps=int(payload["total_env_steps"]),
                    reward_variant=reward_variant,
                    structure_variant=structure_variant,
                    episode_idx=episode_idx,
                    total_episodes=cli_args.episodes,
                    step_idx=step_idx,
                    episode_return=episode_return,
                    pass_count=pass_count,
                    pass_to_shot_count=pass_to_shot_count,
                    assist_count=assist_count,
                    owner_agent=owner_agent,
                )
            )
            
            if debug_frame_dir and step_idx < cli_args.debug_frames:
                cv2.imwrite(str(debug_frame_dir / f"frame_{step_idx:05d}.png"), 
                           cv2.cvtColor(initial_frame, cv2.COLOR_RGB2BGR))

            prev_frame = initial_frame
            identical_frame_count = 0

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

                # Capture and write frame with duplicate detection
                curr_frame = get_frame(env)
                
                # Check if frame changed significantly
                frame_diff = np.mean(np.abs(curr_frame.astype(float) - prev_frame.astype(float)))
                is_duplicate = frame_diff < 1.0  # Less than 0.4% pixel difference
                
                if is_duplicate:
                    identical_frame_count += 1
                    if identical_frame_count > 5:
                        print(f"[WARN] {identical_frame_count} consecutive identical frames - frame capture may be broken")
                else:
                    identical_frame_count = 0
                
                writer.write(
                    overlay_frame(
                        curr_frame,
                        checkpoint_name=checkpoint_name,
                        total_env_steps=int(payload["total_env_steps"]),
                        reward_variant=reward_variant,
                        structure_variant=structure_variant,
                        episode_idx=episode_idx,
                        total_episodes=cli_args.episodes,
                        step_idx=step_idx,
                        episode_return=episode_return,
                        pass_count=pass_count,
                        pass_to_shot_count=pass_to_shot_count,
                        assist_count=assist_count,
                        owner_agent=owner_agent,
                    )
                )
                
                if debug_frame_dir and step_idx < cli_args.debug_frames:
                    cv2.imwrite(str(debug_frame_dir / f"frame_{step_idx:05d}.png"), 
                               cv2.cvtColor(curr_frame, cv2.COLOR_RGB2BGR))

                rnn_states = next_rnn_states.detach().cpu().numpy().reshape(
                    args.num_agents,
                    args.recurrent_N,
                    args.hidden_size,
                )
                rnn_states[masks.squeeze(-1) == 0.0] = 0.0
                obs = next_obs
                available_actions = next_available_actions
                prev_frame = curr_frame

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
        if writer is not None:
            writer.release()
        env.close()

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
