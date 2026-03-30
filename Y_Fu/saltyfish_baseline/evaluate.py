from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from .env import CompetitionEnvConfig, CompetitionRewardConfig, ReducedActionFootballEnv
from .model import SaltyFishModelConfig, StructuredSimple115ActorCritic


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate SaltyFish-inspired single-player baseline.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--compare-random", action="store_true")
    parser.add_argument("--save-video", action="store_true")
    parser.add_argument("--video-dir", default="Y_Fu/videos/saltyfish_baseline")
    return parser.parse_args()


def resolve_device(device_name: str) -> torch.device:
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


def make_env(args: argparse.Namespace, config: dict) -> ReducedActionFootballEnv:
    logdir = None
    other_config_options = {}
    if args.save_video:
        logdir = args.video_dir
        Path(logdir).mkdir(parents=True, exist_ok=True)
        other_config_options["dump_full_episodes"] = True
    env_config = CompetitionEnvConfig(
        env_name=config.get("env_name", "11_vs_11_kaggle"),
        representation=config.get("representation", "simple115v2"),
        rewards=config.get("rewards", "scoring"),
        num_controlled_players=1,
        use_engineered_features=bool(config.get("use_engineered_features", False)),
        reward_config=CompetitionRewardConfig(
            possession_gain_reward=float(config.get("possession_gain_reward", 0.2)),
            possession_loss_penalty=float(config.get("possession_loss_penalty", 0.2)),
            team_possession_reward=float(config.get("team_possession_reward", 0.001)),
            opponent_possession_penalty=float(config.get("opponent_possession_penalty", 0.001)),
            successful_pass_reward=float(config.get("successful_pass_reward", 0.02)),
            progressive_pass_reward_scale=float(config.get("progressive_pass_reward_scale", 0.05)),
            attacking_third_reward=float(config.get("attacking_third_reward", 0.002)),
            shots_with_ball_reward=float(config.get("shots_with_ball_reward", 0.01)),
        ),
    )
    return ReducedActionFootballEnv(
        env_config,
        render=args.render,
        write_video=args.save_video,
        logdir=logdir,
        other_config_options=other_config_options,
    )


def evaluate(env: ReducedActionFootballEnv, model: StructuredSimple115ActorCritic | None, device: torch.device, episodes: int, deterministic: bool) -> dict[str, float]:
    returns: list[float] = []
    goal_diffs: list[float] = []
    lengths: list[int] = []
    win_rates: list[float] = []
    draw_rates: list[float] = []
    goals_for: list[float] = []
    goals_against: list[float] = []
    for episode in range(1, episodes + 1):
        obs = env.reset()
        done = False
        episode_return = 0.0
        episode_length = 0
        while not done:
            if model is None:
                action = env.sample_random_action()
            else:
                obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device)
                with torch.no_grad():
                    action, _, _ = model.act(obs_tensor, deterministic=deterministic)
                action = int(action.item())
            obs, reward, done, _ = env.step(action)
            episode_return += float(np.mean(reward))
            episode_length += 1
        score = env.get_score()
        goal_diff = float(score[0] - score[1]) if score is not None else float("nan")
        win = 1.0 if score is not None and score[0] > score[1] else 0.0
        draw = 1.0 if score is not None and score[0] == score[1] else 0.0
        returns.append(episode_return)
        goal_diffs.append(goal_diff)
        lengths.append(episode_length)
        win_rates.append(win)
        draw_rates.append(draw)
        goals_for.append(float(score[0]) if score is not None else float("nan"))
        goals_against.append(float(score[1]) if score is not None else float("nan"))
        score_text = f"{score[0]}-{score[1]}" if score is not None else "n/a"
        print(f"[episode {episode}] return={episode_return:.3f} score={score_text} goal_diff={goal_diff:.1f} length={episode_length}")
    return {
        "avg_return": float(np.mean(returns)),
        "avg_goal_diff": float(np.mean(goal_diffs)),
        "win_rate": float(np.mean(win_rates)),
        "draw_rate": float(np.mean(draw_rates)),
        "avg_goals_for": float(np.mean(goals_for)),
        "avg_goals_against": float(np.mean(goals_against)),
        "avg_length": float(np.mean(lengths)),
    }


def print_summary(name: str, metrics: dict[str, float]) -> None:
    print(
        f"{name}: avg_return={metrics['avg_return']:.3f} "
        f"avg_goal_diff={metrics['avg_goal_diff']:.3f} "
        f"avg_goals_for={metrics['avg_goals_for']:.3f} "
        f"avg_goals_against={metrics['avg_goals_against']:.3f} "
        f"win_rate={metrics['win_rate']:.3f} "
        f"draw_rate={metrics['draw_rate']:.3f} avg_length={metrics['avg_length']:.1f}"
    )


def main() -> None:
    args = parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = checkpoint.get("config", {})
    env = make_env(args, config)
    model = StructuredSimple115ActorCritic(
        obs_dim=checkpoint["obs_dim"],
        action_dim=checkpoint["action_dim"],
        config=SaltyFishModelConfig(
            head_dim=int(config.get("head_dim", 64)),
            trunk_dim=int(config.get("trunk_dim", 256)),
        ),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    device = resolve_device(args.device)
    model.to(device)
    model.eval()

    trained_metrics = evaluate(env, model, device, args.episodes, args.deterministic)
    env.close()
    print_summary("trained_policy", trained_metrics)

    if args.compare_random:
        random_env = make_env(args, config)
        random_metrics = evaluate(random_env, None, device, args.episodes, args.deterministic)
        random_env.close()
        print_summary("random_policy", random_metrics)


if __name__ == "__main__":
    main()
