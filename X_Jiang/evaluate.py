from __future__ import annotations

import argparse
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from xjiang_football.envs import FootballEnvWrapper
from xjiang_football.legacy_envs import LegacyFootballEnvWrapper
from xjiang_football.legacy_model import LegacyActorCritic
from xjiang_football.legacy_rewards import LegacyRewardShapingConfig
from xjiang_football.model import ActorCritic, ModelConfig
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
            possession_gain_reward=reward_state.get("possession_gain_reward", 0.15),
            possession_loss_penalty=reward_state.get("possession_loss_penalty", 0.15),
            team_possession_reward=reward_state.get("team_possession_reward", 0.001),
            opponent_possession_penalty=reward_state.get("opponent_possession_penalty", 0.001),
            successful_pass_reward=reward_state.get("successful_pass_reward", 0.03),
            progressive_pass_reward_scale=reward_state.get("progressive_pass_reward_scale", 0.08),
            carry_progress_reward_scale=reward_state.get("carry_progress_reward_scale", 0.05),
            attacking_third_reward=reward_state.get("attacking_third_reward", 0.002),
            shot_with_ball_reward=reward_state.get("shot_with_ball_reward", 0.005),
            checkpoint_reward_scale=reward_state.get("checkpoint_reward_scale", 0.03),
            attacking_risk_x_threshold=reward_state.get("attacking_risk_x_threshold", 0.4),
            shot_reward_x_threshold=reward_state.get("shot_reward_x_threshold", 0.35),
            danger_zone_x_threshold=reward_state.get("danger_zone_x_threshold", 0.2),
            danger_zone_entry_reward=reward_state.get("danger_zone_entry_reward", 0.015),
            terminal_zone_x_threshold=reward_state.get("terminal_zone_x_threshold", 0.55),
            terminal_zone_reward=reward_state.get("terminal_zone_reward", 0.02),
            finish_quality_threshold=reward_state.get("finish_quality_threshold", 0.35),
            finish_quality_progress_reward_scale=reward_state.get("finish_quality_progress_reward_scale", 0.04),
            duel_reward_scale=reward_state.get("duel_reward_scale", 0.06),
            low_quality_shot_penalty_scale=reward_state.get("low_quality_shot_penalty_scale", 0.04),
            backtracking_penalty_scale=reward_state.get("backtracking_penalty_scale", 0.04),
            danger_zone_stall_penalty=reward_state.get("danger_zone_stall_penalty", 0.01),
            bad_shot_penalty=reward_state.get("bad_shot_penalty", 0.02),
            attacking_loss_penalty_scale=reward_state.get("attacking_loss_penalty_scale", 0.6),
            out_of_play_loss_penalty=reward_state.get("out_of_play_loss_penalty", 0.05),
        ),
        use_engineered_features=bool(config_dict.get("use_engineered_features", True)),
        collect_feature_metrics=bool(config_dict.get("collect_feature_metrics", False)),
        action_set=str(config_dict.get("action_set", "full")),
        force_shoot_in_zone=bool(config_dict.get("force_shoot_in_zone", False)),
        force_shoot_x_threshold=float(config_dict.get("force_shoot_x_threshold", 0.3)),
        force_shoot_y_threshold=float(config_dict.get("force_shoot_y_threshold", 0.24)),
    )


def build_legacy_env_from_checkpoint(
    checkpoint: dict[str, Any],
    render: bool,
    write_video: bool = False,
    video_dir: str | None = None,
) -> LegacyFootballEnvWrapper:
    config_dict = checkpoint["config"]
    reward_state = checkpoint.get("reward_shaping_state", config_dict)
    logdir = video_dir if write_video and video_dir else config_dict["logdir"]
    if write_video and video_dir:
        Path(video_dir).mkdir(parents=True, exist_ok=True)
    return LegacyFootballEnvWrapper(
        env_name=config_dict["env_name"],
        representation=config_dict["representation"],
        rewards=config_dict["rewards"],
        render=render,
        write_video=write_video,
        logdir=logdir,
        num_controlled_players=config_dict["num_controlled_players"],
        channel_dimensions=tuple(config_dict["channel_dimensions"]),
        reward_shaping=LegacyRewardShapingConfig(
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
        num_players=checkpoint["num_players"],
        config=ModelConfig(
            head_dim=int(config.get("head_dim", 64)),
            trunk_dim=int(config.get("trunk_dim", 256)),
            critic_dim=int(config.get("critic_dim", 256)),
        ),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    model.eval()
    return model


def build_legacy_model_from_checkpoint(checkpoint: dict[str, Any], device: str = "cpu") -> LegacyActorCritic:
    config = checkpoint["config"]
    model = LegacyActorCritic(
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
    video_dir: str = "X_Jiang/videos/five_v_five_football_eval",
) -> None:
    checkpoint = load_checkpoint(checkpoint_path, device=device)
    config_dict = checkpoint["config"]
    is_legacy = int(checkpoint.get("obs_dim", 0)) < 115 or int(checkpoint.get("action_dim", 0)) <= 9
    print(
        f"[startup] loading checkpoint={checkpoint_path} update={checkpoint.get('update', 'n/a')} "
        f"env_name={config_dict['env_name']} players={config_dict['num_controlled_players']} "
        f"mode={'legacy' if is_legacy else 'current'}",
        flush=True,
    )
    if is_legacy:
        env = build_legacy_env_from_checkpoint(checkpoint, render=render, write_video=save_video, video_dir=video_dir)
        model = build_legacy_model_from_checkpoint(checkpoint, device=device)
    else:
        env = build_env_from_checkpoint(checkpoint, render=render, write_video=save_video, video_dir=video_dir)
        model = build_model_from_checkpoint(checkpoint, device=device)

    results = []
    for episode in range(episodes):
        obs = env.reset()
        done = False
        episode_return = 0.0
        steps = 0
        metric_sums: dict[str, float] = defaultdict(float)
        action_counts: dict[str, int] = defaultdict(int)
        episode_start = time.time()
        while not done:
            obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device)
            with torch.no_grad():
                if is_legacy:
                    action_mask_tensor = torch.as_tensor(env.get_action_mask(), dtype=torch.float32, device=device)
                    head_indices_tensor = torch.as_tensor(env.get_policy_head_indices(), dtype=torch.int64, device=device)
                    action, _, _ = model.act(
                        obs_tensor,
                        deterministic=deterministic,
                        action_mask=action_mask_tensor,
                        head_indices=head_indices_tensor,
                    )
                else:
                    action, _, _ = model.act(
                        obs_tensor,
                        deterministic=deterministic,
                    )
            action_np = action.cpu().numpy()
            for action_id in action_np.tolist():
                if is_legacy:
                    action_counts[tactical_action_name(int(action_id))] += 1
                else:
                    action_counts[env.action_name(int(action_id))] += 1

            obs, reward, done, info = env.step(action_np)
            episode_return += float(np.mean(reward))
            steps += 1

            for key, value in info.items():
                if isinstance(value, (int, float)) and not np.isnan(float(value)):
                    metric_sums[key] += float(value)
        score = env.get_score() or (-1, -1)
        result = {
            "return": episode_return,
            "steps": steps,
            "score": score,
            "progression_estimate": metric_sums["progression_estimate"] / float(max(1, steps)),
            "team_spread": metric_sums["team_spread"] / float(max(1, steps)),
            "active_player_distance_to_ball": metric_sums["active_player_distance_to_ball"] / float(max(1, steps)),
            "nearest_teammate_distance": metric_sums["nearest_teammate_distance"] / float(max(1, steps)),
            "nearest_opponent_distance": metric_sums["nearest_opponent_distance"] / float(max(1, steps)),
            "offside_risk": metric_sums["offside_risk"] / float(max(1, steps)),
            "team_possession_rate": metric_sums["team_possession_steps"] / float(max(1, steps * env.num_players)),
            "opponent_possession_rate": metric_sums["opponent_possession_steps"] / float(max(1, steps * env.num_players)),
            "successful_passes": metric_sums["successful_passes"],
            "carry_progress": metric_sums["carry_progress"],
            "shot_actions": metric_sums["shot_actions"],
            "attacking_third_possession_rate": metric_sums["attacking_third_possession_steps"] / float(max(1, steps * env.num_players)),
            "pass_target_space": metric_sums["pass_target_space"] / float(max(1, steps)) if is_legacy else float("nan"),
            "missed_shot_windows": metric_sums["missed_shot_windows"] if is_legacy else float("nan"),
            "tactical_mix": dict(sorted(action_counts.items())),
        }
        results.append(result)
        mix = ",".join(f"{k}={v}" for k, v in result["tactical_mix"].items()) or "n/a"
        legacy_suffix = (
            f" pass_target_space={result['pass_target_space']:.3f} "
            f"missed_shot_windows={result['missed_shot_windows']:.0f} "
            if is_legacy
            else ""
        )
        print(
            f"[episode {episode + 1}] return={episode_return:.3f} steps={steps} score={score[0]}-{score[1]} "
            f"progression={result['progression_estimate']:.3f} "
            f"team_spread={result['team_spread']:.3f} "
            f"ball_dist={result['active_player_distance_to_ball']:.3f} "
            f"nearest_tm={result['nearest_teammate_distance']:.3f} "
            f"nearest_opp={result['nearest_opponent_distance']:.3f} "
            f"offside_risk={result['offside_risk']:.3f} "
            f"team_possession={result['team_possession_rate']:.3f} "
            f"passes={result['successful_passes']:.0f} "
            f"carry_progress={result['carry_progress']:.3f} "
            f"shots={result['shot_actions']:.0f} "
            f"attacking_third={result['attacking_third_possession_rate']:.3f} "
            f"{legacy_suffix}"
            f"action_mix={mix}",
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
    parser.add_argument("--video-dir", type=str, default="X_Jiang/videos/five_v_five_football_eval")
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
