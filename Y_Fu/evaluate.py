from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import torch

from yfu_football.envs import FootballEnvWrapper, RewardShapingConfig
from yfu_football.model import ActorCritic


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a saved PPO checkpoint on Google Research Football.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--compare-random", action="store_true")
    parser.add_argument("--save-video", action="store_true")
    parser.add_argument("--video-dir", default="Y_Fu/videos")
    parser.add_argument("--stop-on-success", action="store_true")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--env-name")
    parser.add_argument("--representation")
    parser.add_argument("--rewards")
    parser.add_argument("--num-controlled-players", type=int)
    parser.add_argument("--channel-width", type=int)
    parser.add_argument("--channel-height", type=int)
    return parser.parse_args()


def resolve_device(device_name: str) -> torch.device:
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


def resolve_channel_dimensions(args: argparse.Namespace, config: dict[str, Any]) -> tuple[int, int]:
    if args.channel_width is not None and args.channel_height is not None:
        return args.channel_width, args.channel_height
    if "channel_dimensions" in config:
        width, height = config["channel_dimensions"]
        return int(width), int(height)
    return (42, 42)


def set_global_seed(seed: int | None) -> None:
    if seed is None:
        return
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_env(args: argparse.Namespace, config: dict[str, Any]) -> FootballEnvWrapper:
    other_config_options: dict[str, Any] = {}
    logdir = None
    if args.save_video:
        logdir = args.video_dir
        Path(logdir).mkdir(parents=True, exist_ok=True)
        other_config_options["dump_full_episodes"] = True
    if args.seed is not None:
        other_config_options["game_engine_random_seed"] = int(args.seed)

    return FootballEnvWrapper(
        env_name=args.env_name or config.get("env_name", "11_vs_11_easy_stochastic"),
        representation=args.representation or config.get("representation", "simple115v2"),
        rewards=args.rewards or config.get("rewards", "scoring,checkpoints"),
        render=args.render,
        write_video=args.save_video,
        logdir=logdir,
        num_controlled_players=args.num_controlled_players or config.get("num_controlled_players", 11),
        channel_dimensions=resolve_channel_dimensions(args, config),
        other_config_options=other_config_options,
        reward_shaping=RewardShapingConfig(
            pass_success_reward=float(config.get("pass_success_reward", 0.0)),
            pass_failure_penalty=float(config.get("pass_failure_penalty", 0.0)),
            pass_progress_reward_scale=float(config.get("pass_progress_reward_scale", 0.0)),
            shot_attempt_reward=float(config.get("shot_attempt_reward", 0.0)),
            attacking_possession_reward=float(config.get("attacking_possession_reward", 0.0)),
            attacking_x_threshold=float(config.get("attacking_x_threshold", 0.55)),
            final_third_entry_reward=float(config.get("final_third_entry_reward", 0.0)),
            possession_retention_reward=float(config.get("possession_retention_reward", 0.0)),
            own_half_turnover_penalty=float(config.get("own_half_turnover_penalty", 0.0)),
            own_half_x_threshold=float(config.get("own_half_x_threshold", 0.0)),
            pending_pass_horizon=int(config.get("pending_pass_horizon", 8)),
        ),
    )


def evaluate_agent(
    env: FootballEnvWrapper,
    episodes: int,
    actor: ActorCritic | None,
    device: torch.device,
    deterministic: bool,
    stop_on_success: bool = False,
) -> dict[str, float]:
    episode_returns: list[float] = []
    episode_lengths: list[int] = []
    score_rewards: list[float] = []
    goal_diffs: list[float] = []
    win_indicators: list[float] = []

    for episode_idx in range(1, episodes + 1):
        observation = env.reset()
        done = False
        episode_return = 0.0
        episode_length = 0
        total_score_reward = 0.0

        while not done:
            if actor is None:
                actions = env.sample_random_action()
            else:
                observation_tensor = torch.as_tensor(observation, dtype=torch.float32, device=device)
                with torch.no_grad():
                    sampled_actions, _, _ = actor.act(observation_tensor, deterministic=deterministic)
                actions = sampled_actions.cpu().numpy()

            observation, reward, done, info = env.step(actions)
            episode_return += float(np.mean(reward))
            total_score_reward += float(info.get("score_reward", 0.0))
            episode_length += 1

        score = env.get_score()
        goal_diff = float(score[0] - score[1]) if score is not None else float("nan")
        win = 1.0 if score is not None and score[0] > score[1] else 0.0
        success = win > 0.0 or total_score_reward > 0.0

        episode_returns.append(episode_return)
        episode_lengths.append(episode_length)
        score_rewards.append(total_score_reward)
        goal_diffs.append(goal_diff)
        win_indicators.append(win)

        score_text = f"{score[0]}-{score[1]}" if score is not None else "n/a"
        print(
            f"[episode {episode_idx}] "
            f"return={episode_return:.3f} "
            f"score_reward={total_score_reward:.3f} "
            f"score={score_text} "
            f"goal_diff={goal_diff:.1f} "
            f"length={episode_length}"
        )

        if stop_on_success and success:
            print(f"success_episode: {episode_idx}")
            break

    return {
        "avg_return": float(np.mean(episode_returns)),
        "avg_score_reward": float(np.mean(score_rewards)),
        "avg_goal_diff": float(np.mean(goal_diffs)),
        "win_rate": float(np.mean(win_indicators)),
        "avg_length": float(np.mean(episode_lengths)),
    }


def print_summary(name: str, metrics: dict[str, float]) -> None:
    print(
        f"{name}: "
        f"avg_return={metrics['avg_return']:.3f} "
        f"avg_score_reward={metrics['avg_score_reward']:.3f} "
        f"avg_goal_diff={metrics['avg_goal_diff']:.3f} "
        f"win_rate={metrics['win_rate']:.3f} "
        f"avg_length={metrics['avg_length']:.1f}"
    )


def main() -> None:
    args = parse_args()
    set_global_seed(args.seed)
    checkpoint_path = Path(args.checkpoint)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    config = checkpoint.get("config", {})

    env = make_env(args, config)
    model = ActorCritic(
        obs_dim=checkpoint["obs_dim"],
        action_dim=checkpoint["action_dim"],
        hidden_sizes=tuple(config.get("hidden_sizes", [256, 256])),
        obs_shape=tuple(checkpoint.get("obs_shape", (checkpoint["obs_dim"],))),
        model_type=config.get("model_type", "auto"),
        feature_dim=int(config.get("feature_dim", 256)),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    device = resolve_device(args.device)
    model.to(device)
    model.eval()

    trained_metrics = evaluate_agent(
        env=env,
        episodes=args.episodes,
        actor=model,
        device=device,
        deterministic=args.deterministic,
        stop_on_success=args.stop_on_success,
    )
    env.close()
    print_summary("trained_policy", trained_metrics)
    if args.save_video:
        print(f"video_output_dir: {args.video_dir}")

    if args.compare_random:
        random_env = make_env(args, config)
        random_metrics = evaluate_agent(
            env=random_env,
            episodes=args.episodes,
            actor=None,
            device=device,
            deterministic=args.deterministic,
            stop_on_success=args.stop_on_success,
        )
        random_env.close()
        print_summary("random_policy", random_metrics)
        print(
            "delta_vs_random: "
            f"avg_return={trained_metrics['avg_return'] - random_metrics['avg_return']:.3f} "
            f"avg_score_reward={trained_metrics['avg_score_reward'] - random_metrics['avg_score_reward']:.3f} "
            f"avg_goal_diff={trained_metrics['avg_goal_diff'] - random_metrics['avg_goal_diff']:.3f} "
            f"win_rate={trained_metrics['win_rate'] - random_metrics['win_rate']:.3f}"
        )


if __name__ == "__main__":
    main()
