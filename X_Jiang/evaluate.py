from __future__ import annotations

import argparse
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from xjiang_football.envs import FootballEnvWrapper
from xjiang_football.model import ActorCritic
from xjiang_football.rewards import RewardShapingConfig
from xjiang_football.utils import tactical_action_name


def load_checkpoint(checkpoint_path: str, device: str = "cpu") -> dict[str, Any]:
    return torch.load(checkpoint_path, map_location=device)


def build_env_from_checkpoint(
    checkpoint: dict[str, Any],
    render: bool,
    write_video: bool = False,
    video_dir: str | None = None,
) -> FootballEnvWrapper:
    config_dict = checkpoint["config"]
    reward_state = checkpoint.get("reward_shaping_state", config_dict)
    logdir = video_dir if write_video and video_dir else config_dict["logdir"]
    if write_video and video_dir:
        Path(video_dir).mkdir(parents=True, exist_ok=True)
    return FootballEnvWrapper(
        env_name=config_dict["env_name"],
        representation=config_dict["representation"],
        rewards=config_dict["rewards"],
        render=render,
        write_video=write_video,
        logdir=logdir,
        num_controlled_players=config_dict["num_controlled_players"],
        channel_dimensions=tuple(config_dict["channel_dimensions"]),
        reward_shaping=RewardShapingConfig(
            enabled=reward_state.get("enabled", config_dict.get("use_reward_shaping", True)),
            closest_player_to_ball_reward=reward_state.get("closest_player_to_ball_reward", 0.08),
            first_defender_pressure_reward=reward_state.get("first_defender_pressure_reward", 0.06),
            second_player_support_reward=reward_state.get("second_player_support_reward", 0.05),
            recover_shape_reward=reward_state.get("recover_shape_reward", 0.04),
            hold_shape_reward=reward_state.get("hold_shape_reward", 0.02),
            ball_watch_penalty=reward_state.get("ball_watch_penalty", 0.06),
            idle_wander_penalty=reward_state.get("idle_wander_penalty", 0.03),
            goalkeeper_home_reward=reward_state.get("goalkeeper_home_reward", 0.03),
            goalkeeper_wander_penalty=reward_state.get("goalkeeper_wander_penalty", 0.08),
            possession_support_reward=reward_state.get("possession_support_reward", 0.03),
            attack_space_reward=reward_state.get("attack_space_reward", 0.04),
            progressive_pass_choice_reward=reward_state.get("progressive_pass_choice_reward", 0.05),
            progressive_pass_result_reward_scale=reward_state.get("progressive_pass_result_reward_scale", 0.10),
            carry_progress_reward_scale=reward_state.get("carry_progress_reward_scale", 0.06),
            zone_entry_progress_reward=reward_state.get("zone_entry_progress_reward", 0.10),
            safe_reset_pass_reward=reward_state.get("safe_reset_pass_reward", 0.01),
            backward_gk_pass_penalty=reward_state.get("backward_gk_pass_penalty", 0.08),
            unnecessary_goalkeeper_reset_penalty=reward_state.get("unnecessary_goalkeeper_reset_penalty", 0.12),
            non_emergency_clear_penalty=reward_state.get("non_emergency_clear_penalty", 0.10),
            shot_choice_reward=reward_state.get("shot_choice_reward", 0.05),
            missed_shot_window_penalty=reward_state.get("missed_shot_window_penalty", 0.10),
            shot_execution_reward=reward_state.get("shot_execution_reward", 0.15),
            on_ball_stall_penalty=reward_state.get("on_ball_stall_penalty", 0.08),
            on_ball_backward_drift_penalty=reward_state.get("on_ball_backward_drift_penalty", 0.07),
            on_ball_lateral_zigzag_penalty=reward_state.get("on_ball_lateral_zigzag_penalty", 0.05),
            support_spacing_reward=reward_state.get("support_spacing_reward", 0.05),
            support_spacing_penalty=reward_state.get("support_spacing_penalty", 0.04),
            support_forward_lane_reward=reward_state.get("support_forward_lane_reward", 0.04),
            support_static_penalty=reward_state.get("support_static_penalty", 0.03),
            safe_reset_overuse_penalty=reward_state.get("safe_reset_overuse_penalty", 0.10),
            support_behind_ball_penalty=reward_state.get("support_behind_ball_penalty", 0.05),
        ),
    )


def build_model_from_checkpoint(checkpoint: dict[str, Any], device: str = "cpu") -> ActorCritic:
    config = checkpoint["config"]
    model = ActorCritic(
        obs_dim=checkpoint["obs_dim"],
        action_dim=checkpoint["action_dim"],
        hidden_sizes=tuple(config["hidden_sizes"]),
        obs_shape=tuple(checkpoint["obs_shape"]),
        model_type=config["model_type"],
        feature_dim=config["feature_dim"],
        use_specialized_policy_heads=bool(config.get("use_specialized_policy_heads", False)),
        use_specialized_value_heads=bool(config.get("use_specialized_value_heads", False)),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    model.eval()
    return model


def evaluate(
    checkpoint_path: str,
    episodes: int = 1,
    deterministic: bool = True,
    render: bool = True,
    device: str = "cpu",
    save_video: bool = False,
    video_dir: str = "X_Jiang/videos/five_v_five_tactical_eval",
) -> None:
    checkpoint = load_checkpoint(checkpoint_path, device=device)
    config_dict = checkpoint["config"]
    print(
        f"[startup] loading checkpoint={checkpoint_path} update={checkpoint.get('update', 'n/a')} "
        f"env_name={config_dict['env_name']} players={config_dict['num_controlled_players']} model={config_dict['model_type']}",
        flush=True,
    )
    env = build_env_from_checkpoint(checkpoint, render=render, write_video=save_video, video_dir=video_dir)
    model = build_model_from_checkpoint(checkpoint, device=device)
    print(f"[startup] evaluation roles={env.get_role_names()}", flush=True)

    results = []
    for episode in range(episodes):
        obs = env.reset()
        done = False
        episode_return = 0.0
        steps = 0
        metric_lists: dict[str, list[float]] = defaultdict(list)
        tactical_counts: dict[str, int] = defaultdict(int)
        episode_start = time.time()

        while not done:
            obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device)
            action_mask_tensor = torch.as_tensor(env.get_action_mask(), dtype=torch.float32, device=device)
            head_indices_tensor = torch.as_tensor(env.get_policy_head_indices(), dtype=torch.int64, device=device)
            with torch.no_grad():
                action, _, _ = model.act(
                    obs_tensor,
                    deterministic=deterministic,
                    action_mask=action_mask_tensor,
                    head_indices=head_indices_tensor,
                )
            action_np = action.cpu().numpy()
            for action_id in action_np.tolist():
                tactical_counts[tactical_action_name(int(action_id))] += 1

            obs, reward, done, info = env.step(action_np)
            episode_return += float(np.mean(reward))
            steps += 1

            for key in (
                "goalkeeper_x",
                "active_player_distance_to_ball",
                "closest_outfield_distance_to_ball",
                "second_outfield_distance_to_ball",
                "team_spread",
                "progression_estimate",
                "pass_target_space",
                "missed_shot_windows",
                "gk_reset_events",
                "free_ball_attack_phase",
            ):
                value = float(info.get(key, np.nan))
                if not np.isnan(value):
                    metric_lists[key].append(value)

            if steps % 200 == 0:
                print(
                    f"[episode {episode + 1}] progress steps={steps} "
                    f"elapsed={time.time() - episode_start:.1f}s partial_return={episode_return:.3f}",
                    flush=True,
                )

        score = env.get_score() or (-1, -1)
        result = {
            "return": episode_return,
            "steps": steps,
            "score": score,
            "goalkeeper_avg_x": float(np.mean(metric_lists.get("goalkeeper_x", [np.nan]))),
            "closest_outfield_ball_distance": float(np.mean(metric_lists.get("closest_outfield_distance_to_ball", [np.nan]))),
            "second_outfield_ball_distance": float(np.mean(metric_lists.get("second_outfield_distance_to_ball", [np.nan]))),
            "team_spread": float(np.mean(metric_lists.get("team_spread", [np.nan]))),
            "progression_estimate": float(np.mean(metric_lists.get("progression_estimate", [0.0]))),
            "pass_target_space": float(np.mean(metric_lists.get("pass_target_space", [np.nan]))),
            "missed_shot_windows": float(np.sum(metric_lists.get("missed_shot_windows", [0.0]))),
            "gk_reset_events": float(np.sum(metric_lists.get("gk_reset_events", [0.0]))),
            "free_ball_attack_phase_rate": float(np.mean(metric_lists.get("free_ball_attack_phase", [0.0]))),
            "tactical_mix": dict(sorted(tactical_counts.items())),
        }
        results.append(result)
        mix = ",".join(f"{k}={v}" for k, v in result["tactical_mix"].items()) or "n/a"
        print(
            f"[episode {episode + 1}] return={episode_return:.3f} steps={steps} score={score[0]}-{score[1]} "
            f"goalkeeper_avg_x={result['goalkeeper_avg_x']:.3f} "
            f"closest_outfield_ball_dist={result['closest_outfield_ball_distance']:.3f} "
            f"second_outfield_ball_dist={result['second_outfield_ball_distance']:.3f} "
            f"team_spread={result['team_spread']:.3f} progression={result['progression_estimate']:.3f} "
            f"pass_target_space={result['pass_target_space']:.3f} "
            f"missed_shot_windows={result['missed_shot_windows']:.0f} "
            f"gk_reset_events={result['gk_reset_events']:.0f} "
            f"free_ball_attack_rate={result['free_ball_attack_phase_rate']:.3f} "
            f"tactical_mix={mix}",
            flush=True,
        )

    env.close()
    if results:
        avg_return = sum(r["return"] for r in results) / len(results)
        avg_goals_for = sum(r["score"][0] for r in results) / len(results)
        avg_goals_against = sum(r["score"][1] for r in results) / len(results)
        print("\n=== Summary ===")
        print(f"episodes: {len(results)}")
        print(f"avg_return: {avg_return:.3f}")
        print(f"avg_goals_for: {avg_goals_for:.2f}")
        print(f"avg_goals_against: {avg_goals_against:.2f}")
        if save_video:
            print(f"video_output_dir: {video_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate X_Jiang football model.")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--stochastic", action="store_true")
    parser.add_argument("--no-render", action="store_true")
    parser.add_argument("--save-video", action="store_true")
    parser.add_argument("--video-dir", type=str, default="X_Jiang/videos/five_v_five_tactical_eval")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate(
        checkpoint_path=args.checkpoint,
        episodes=args.episodes,
        deterministic=not args.stochastic,
        render=not args.no_render,
        device=args.device,
        save_video=args.save_video,
        video_dir=args.video_dir,
    )
