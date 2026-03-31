from __future__ import annotations

import argparse
from pathlib import Path

import torch

from presets import TrainConfig
from xjiang_football.envs import FootballEnvWrapper
from xjiang_football.model import ActorCritic


def load_checkpoint(checkpoint_path: str, device: str = "cpu"):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    return checkpoint


def build_env_from_checkpoint(
    config_dict: dict,
    render: bool,
    write_video: bool = False,
    video_dir: str | None = None,
) -> FootballEnvWrapper:
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
    )


def build_model_from_checkpoint(checkpoint: dict, device: str = "cpu") -> ActorCritic:
    config = checkpoint["config"]
    model = ActorCritic(
        obs_dim=checkpoint["obs_dim"],
        action_dim=checkpoint["action_dim"],
        hidden_sizes=tuple(config["hidden_sizes"]),
        obs_shape=tuple(checkpoint["obs_shape"]),
        model_type=config["model_type"],
        feature_dim=config["feature_dim"],
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def evaluate(
    checkpoint_path: str,
    episodes: int = 1,
    deterministic: bool = True,
    render: bool = True,
    device: str = "cpu",
    save_video: bool = False,
    video_dir: str = "X_Jiang/videos/five_v_five_debug",
) -> None:
    checkpoint = load_checkpoint(checkpoint_path, device=device)
    config_dict = checkpoint["config"]

    env = build_env_from_checkpoint(
        config_dict,
        render=render,
        write_video=save_video,
        video_dir=video_dir,
    )
    model = build_model_from_checkpoint(checkpoint, device=device)

    results = []

    for episode in range(episodes):
        obs = env.reset()
        done = False
        episode_return = 0.0
        steps = 0

        while not done:
            obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device)
            with torch.no_grad():
                action, _, _ = model.act(obs_tensor, deterministic=deterministic)

            action_np = action.cpu().numpy()
            obs, reward, done, info = env.step(action_np)

            episode_return += float(reward.mean())
            steps += 1

        score = env.get_score()
        if score is None:
            score = (-1, -1)

        results.append(
            {
                "episode": episode + 1,
                "return": episode_return,
                "steps": steps,
                "score": score,
                "score_reward": float(info.get("score_reward", 0.0)),
            }
        )

        print(
            f"[episode {episode + 1}] "
            f"return={episode_return:.3f} "
            f"steps={steps} "
            f"score={score[0]}-{score[1]} "
            f"score_reward={float(info.get('score_reward', 0.0)):.3f}"
        )

    env.close()

    if results:
        avg_return = sum(r["return"] for r in results) / len(results)
        avg_steps = sum(r["steps"] for r in results) / len(results)
        avg_goals_for = sum(r["score"][0] for r in results) / len(results)
        avg_goals_against = sum(r["score"][1] for r in results) / len(results)

        print("\n=== Summary ===")
        print(f"episodes: {len(results)}")
        print(f"avg_return: {avg_return:.3f}")
        print(f"avg_steps: {avg_steps:.1f}")
        print(f"avg_goals_for: {avg_goals_for:.2f}")
        print(f"avg_goals_against: {avg_goals_against:.2f}")
        if save_video:
            print(f"video_output_dir: {video_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate X_Jiang football model.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to checkpoint, e.g. X_Jiang/checkpoints/five_v_five_debug/latest.pt",
    )
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--stochastic", action="store_true")
    parser.add_argument("--no-render", action="store_true")
    parser.add_argument("--save-video", action="store_true")
    parser.add_argument("--video-dir", type=str, default="X_Jiang/videos/five_v_five_debug")
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
