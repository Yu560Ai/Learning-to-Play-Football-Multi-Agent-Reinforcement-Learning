#!/usr/bin/env python3
"""Debug version of render_policy_video.py that checks for environment progression."""

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
        description="Debug render with detailed frame and environment checks.",
    )
    parser.add_argument("--checkpoint", default=None, help="Path to the checkpoint .pt file.")
    parser.add_argument("--untrained", action="store_true", default=False, help="Use random init policy.")
    parser.add_argument("--output_mp4", required=True, help="Output video path.")
    parser.add_argument("--episodes", type=int, default=1, help="Number of episodes to record.")
    parser.add_argument("--fps", type=float, default=10.0, help="Output video framerate.")
    parser.add_argument("--seed", type=int, default=7, help="Game engine seed.")
    parser.add_argument("--max_steps", type=int, default=None, help="Per-episode cap.")
    parser.add_argument("--debug_steps", type=int, default=20, help="Number of steps to debug with verbose output.")
    parser.add_argument("--policy_mode", type=str, default="trained", choices=["trained", "random", "fixed"],
                        help="Which policy to use: trained checkpoint, random actions, or fixed action.")
    return parser


def load_checkpoint_args(checkpoint_path: Path) -> tuple[Namespace, dict]:
    payload = torch.load(checkpoint_path, map_location="cpu")
    args = Namespace(**payload["args"])
    args.disable_cuda = True
    args.render = False
    args.save_replay = False
    args.n_rollout_threads = 1
    return args, payload


def build_untrained_args() -> Namespace:
    args = build_train_parser().parse_args([])
    args.disable_cuda = True
    args.render = False
    args.save_replay = False
    args.n_rollout_threads = 1
    return args


def get_frame(env: TwoVTwoFootballEnv) -> np.ndarray:
    targets = [getattr(env, "env", None), getattr(getattr(env, "env", None), "unwrapped", None)]
    for target in targets:
        if target is None:
            continue
        try:
            frame = target.render(mode="rgb_array")
        except Exception as e:
            print(f"  [WARNING] Frame capture failed: {e}")
            continue
        if frame is not None:
            return np.asarray(frame, dtype=np.uint8)
    raise RuntimeError("Could not capture an RGB frame from the football environment.")


def frames_are_identical(frame1: np.ndarray, frame2: np.ndarray) -> tuple[bool, float]:
    """Check if two frames are identical. Return (is_identical, percent_diff)."""
    if frame1.shape != frame2.shape:
        return False, 100.0
    
    diff = np.abs(frame1.astype(float) - frame2.astype(float))
    percent_diff = 100.0 * np.mean(diff) / 255.0
    
    # Frames are identical if less than 0.1% different
    is_identical = percent_diff < 0.1
    return is_identical, percent_diff


def analyze_ball_position(raw_obs: list[dict]) -> tuple[float, float, float]:
    """Extract ball position (x, y, z) from raw observation."""
    if not raw_obs:
        return 0.0, 0.0, 0.0
    ball = raw_obs[0].get("ball", (0.0, 0.0, 0.0))
    return float(ball[0]), float(ball[1]), float(ball[2])


def analyze_player_positions(raw_obs: list[dict], agent_idx: int) -> tuple[float, float]:
    """Extract controlled player position for given agent."""
    if agent_idx >= len(raw_obs):
        return 0.0, 0.0
    obs = raw_obs[agent_idx]
    players_control = obs.get("players_on_pitch", [])
    if not players_control or len(players_control) == 0:
        return 0.0, 0.0
    # Position of controlled player (typically first element)
    pos = players_control[0]
    if not pos or len(pos) < 2:
        return 0.0, 0.0
    return float(pos[0]), float(pos[1])


def main() -> None:
    cli_args = build_parser().parse_args()
    output_mp4 = Path(cli_args.output_mp4).resolve()
    output_mp4.parent.mkdir(parents=True, exist_ok=True)

    checkpoint_path: Path | None = None

    if cli_args.untrained:
        args = build_untrained_args()
        payload = {
            "update": 0,
            "total_env_steps": 0,
            "actor_state_dict": None,
        }
        checkpoint_name = "untrained"
    else:
        if cli_args.checkpoint is None:
            raise ValueError("Either --checkpoint PATH or --untrained must be provided.")
        checkpoint_path = Path(cli_args.checkpoint).resolve()
        args, payload = load_checkpoint_args(checkpoint_path)
        checkpoint_name = checkpoint_path.name

    args.game_engine_random_seed = int(cli_args.seed)

    # Create environment
    print(f"[DEBUG] Creating environment with scenario={args.scenario_name}, seed={cli_args.seed}")
    env = TwoVTwoFootballEnv(args, rank=0, log_dir=None, is_eval=True)
    print(f"[DEBUG] Environment created. num_agents={env.num_agents}, action_dim={env.action_dim}")

    # Load policy
    obs_space = env.observation_space[0]
    share_obs_space = env.share_observation_space[0]
    act_space = env.action_space[0]

    module = MAPPOModule(args, obs_space, share_obs_space, act_space, device=torch.device("cpu"))
    if payload["actor_state_dict"] is not None:
        module.actor.load_state_dict(payload["actor_state_dict"])
        print(f"[DEBUG] Loaded checkpoint actor weights")
    module.actor.eval()

    # Prepare video writer (lazy init)
    writer: cv2.VideoWriter | None = None
    max_steps = int(cli_args.max_steps) if cli_args.max_steps is not None else int(args.episode_length)
    debug_steps = min(int(cli_args.debug_steps), max_steps)

    print(f"\n[DEBUG] Starting {cli_args.episodes} episodes with {max_steps} max steps/episode")
    print(f"[DEBUG] Will print detailed debug info for first {debug_steps} steps")
    print(f"[DEBUG] Policy mode: {cli_args.policy_mode}\n")

    try:
        for episode_idx in range(1, cli_args.episodes + 1):
            print(f"\n{'='*80}")
            print(f"EPISODE {episode_idx}/{cli_args.episodes}")
            print(f"{'='*80}")

            # Reset environment
            print(f"[RESET] env.reset()")
            obs, _share_obs, available_actions = env.reset()
            prev_raw_obs = env._raw_observation()
            print(f"[RESET] Initial observation shape: {obs.shape}")
            print(f"[RESET] Available actions shape: {available_actions.shape}")
            print(f"[RESET] Ball position: {analyze_ball_position(prev_raw_obs)}")

            rnn_states = np.zeros((args.num_agents, args.recurrent_N, args.hidden_size), dtype=np.float32)
            masks = np.ones((args.num_agents, 1), dtype=np.float32)
            done = False
            step_idx = 0
            episode_return = 0.0
            prev_frame = None

            # Get initial frame
            try:
                initial_frame = get_frame(env)
                if writer is None:
                    height, width = initial_frame.shape[:2]
                    output_mp4.parent.mkdir(parents=True, exist_ok=True)
                    writer = cv2.VideoWriter(
                        str(output_mp4),
                        cv2.VideoWriter_fourcc(*"mp4v"),
                        cli_args.fps,
                        (width, height),
                    )
                    if not writer.isOpened():
                        raise RuntimeError(f"Could not open video writer for {output_mp4}.")
                    print(f"[VIDEO] Initialized MP4 writer: {width}x{height}, {cli_args.fps} fps")
                
                frame_bgr = cv2.cvtColor(initial_frame, cv2.COLOR_RGB2BGR)
                writer.write(frame_bgr)
                prev_frame = initial_frame
                print(f"[FRAME:0] Wrote initial frame")
            except Exception as e:
                print(f"[ERROR] Failed to get initial frame: {e}")
                continue

            # Main episode loop
            while not done and step_idx < max_steps:
                step_idx += 1
                is_debug_step = step_idx <= debug_steps

                if is_debug_step:
                    print(f"\n--- STEP {step_idx} ---")

                # Get action
                if cli_args.policy_mode == "random":
                    env_actions = np.random.randint(0, env.action_dim, size=(env.num_agents,), dtype=np.int64)
                    if is_debug_step:
                        print(f"[ACTION] Mode: RANDOM, Actions: {env_actions}")
                elif cli_args.policy_mode == "fixed":
                    env_actions = np.array([0] * env.num_agents, dtype=np.int64)
                    if is_debug_step:
                        print(f"[ACTION] Mode: FIXED (0), Actions: {env_actions}")
                else:
                    # Use trained policy
                    with torch.no_grad():
                        actions, _action_log_probs, next_rnn_states = module.actor(
                            obs.reshape(-1, obs.shape[-1]),
                            rnn_states.reshape(-1, *rnn_states.shape[1:]),
                            masks.reshape(-1, 1),
                            available_actions.reshape(-1, available_actions.shape[-1]),
                            deterministic=True,
                        )
                    env_actions = actions.detach().cpu().numpy().reshape(args.num_agents, -1).squeeze(-1).astype(np.int64)
                    if is_debug_step:
                        print(f"[ACTION] Mode: TRAINED, Actions: {env_actions}")

                # Step environment
                if is_debug_step:
                    print(f"[BEFORE_STEP] Ball: {analyze_ball_position(prev_raw_obs)}")
                    print(f"[BEFORE_STEP] Player 0 pos: {analyze_player_positions(prev_raw_obs, 0)}")
                    print(f"[BEFORE_STEP] Player 1 pos: {analyze_player_positions(prev_raw_obs, 1)}")

                next_obs, _next_share_obs, rewards, dones, infos, next_available_actions = env.step(env_actions)
                curr_raw_obs = env._raw_observation()

                if is_debug_step:
                    print(f"[AFTER_STEP] Reward: {rewards.T}")
                    print(f"[AFTER_STEP] Done: {dones}")
                    print(f"[AFTER_STEP] Ball: {analyze_ball_position(curr_raw_obs)}")
                    print(f"[AFTER_STEP] Player 0 pos: {analyze_player_positions(curr_raw_obs, 0)}")
                    print(f"[AFTER_STEP] Player 1 pos: {analyze_player_positions(curr_raw_obs, 1)}")
                    
                    obs_changed = not np.allclose(obs, next_obs, atol=1e-6)
                    print(f"[OBS] Changed: {obs_changed}, shape={next_obs.shape}")

                episode_return += float(np.mean(rewards))
                done = bool(np.all(dones))

                # Capture frame
                try:
                    curr_frame = get_frame(env)
                    
                    if is_debug_step:
                        identical, pct_diff = frames_are_identical(prev_frame, curr_frame)
                        print(f"[FRAME:{step_idx}] Identical to prev: {identical}, diff: {pct_diff:.2f}%")
                    
                    frame_bgr = cv2.cvtColor(curr_frame, cv2.COLOR_RGB2BGR)
                    writer.write(frame_bgr)
                    prev_frame = curr_frame
                except Exception as e:
                    print(f"[ERROR] Failed to capture/write frame at step {step_idx}: {e}")

                # Update state
                if cli_args.policy_mode == "trained":
                    rnn_states = next_rnn_states.detach().cpu().numpy().reshape(
                        args.num_agents,
                        args.recurrent_N,
                        args.hidden_size,
                    )
                    rnn_states[masks.squeeze(-1) == 0.0] = 0.0
                
                obs = next_obs
                prev_raw_obs = curr_raw_obs
                available_actions = next_available_actions

                if done or step_idx >= max_steps:
                    print(f"\n[EPISODE_END] Final step: {step_idx}, Return: {episode_return:.3f}")

    finally:
        if writer is not None:
            writer.release()
            print(f"\n[VIDEO] Released writer, saved to: {output_mp4}")
        env.close()
        print(f"[COMPLETE] Environment closed")


if __name__ == "__main__":
    main()
