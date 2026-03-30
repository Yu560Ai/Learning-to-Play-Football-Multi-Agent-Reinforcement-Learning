from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np

from .env import ArenaMatchEnv
from .registry import create_agent


@dataclass
class EpisodeResult:
    left_reward: float
    right_reward: float
    left_score: int
    right_score: int
    length: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a head-to-head football arena match between two local agents.")
    parser.add_argument("--env-name", default="11_vs_11_kaggle")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--left-agent", default="yfu_saltyfish")
    parser.add_argument("--right-agent", default="random")
    parser.add_argument("--left-checkpoint")
    parser.add_argument("--right-checkpoint")
    parser.add_argument("--left-device", default="cpu")
    parser.add_argument("--right-device", default="cpu")
    parser.add_argument("--deterministic-left", action="store_true")
    parser.add_argument("--deterministic-right", action="store_true")
    parser.add_argument("--left-seed", type=int)
    parser.add_argument("--right-seed", type=int)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--save-video", action="store_true")
    parser.add_argument("--video-dir", default="Arena/videos")
    return parser.parse_args()


def summarize(name: str, results: list[EpisodeResult]) -> None:
    left_rewards = np.asarray([r.left_reward for r in results], dtype=np.float32)
    right_rewards = np.asarray([r.right_reward for r in results], dtype=np.float32)
    left_goals = np.asarray([r.left_score for r in results], dtype=np.float32)
    right_goals = np.asarray([r.right_score for r in results], dtype=np.float32)
    lengths = np.asarray([r.length for r in results], dtype=np.float32)
    left_wins = np.asarray([1.0 if r.left_score > r.right_score else 0.0 for r in results], dtype=np.float32)
    draws = np.asarray([1.0 if r.left_score == r.right_score else 0.0 for r in results], dtype=np.float32)
    print(
        f"{name}: left_avg_reward={left_rewards.mean():.3f} "
        f"right_avg_reward={right_rewards.mean():.3f} "
        f"avg_score={left_goals.mean():.3f}-{right_goals.mean():.3f} "
        f"avg_goal_diff={(left_goals - right_goals).mean():.3f} "
        f"left_win_rate={left_wins.mean():.3f} draw_rate={draws.mean():.3f} "
        f"avg_length={lengths.mean():.1f}"
    )


def main() -> None:
    args = parse_args()
    env = ArenaMatchEnv(
        env_name=args.env_name,
        render=args.render,
        save_video=args.save_video,
        video_dir=args.video_dir,
    )
    left_agent = create_agent(
        args.left_agent,
        checkpoint=args.left_checkpoint,
        device=args.left_device,
        num_actions=env.num_actions,
        seed=args.left_seed,
    )
    right_agent = create_agent(
        args.right_agent,
        checkpoint=args.right_checkpoint,
        device=args.right_device,
        num_actions=env.num_actions,
        seed=args.right_seed,
    )

    results: list[EpisodeResult] = []
    try:
        for episode in range(1, args.episodes + 1):
            left_obs, right_obs = env.reset()
            left_agent.reset()
            right_agent.reset()
            left_reward_total = 0.0
            right_reward_total = 0.0
            steps = 0
            done = False

            while not done:
                left_action = left_agent.act(left_obs, deterministic=args.deterministic_left)
                right_action = right_agent.act(right_obs, deterministic=args.deterministic_right)
                left_obs, right_obs, rewards, done, _ = env.step(left_action, right_action)
                left_reward_total += float(rewards[0])
                right_reward_total += float(rewards[1])
                steps += 1
                if args.max_steps is not None and steps >= args.max_steps:
                    done = True

            left_score, right_score = env.get_score()
            results.append(
                EpisodeResult(
                    left_reward=left_reward_total,
                    right_reward=right_reward_total,
                    left_score=left_score,
                    right_score=right_score,
                    length=steps,
                )
            )
            print(
                f"[episode {episode}] left_reward={left_reward_total:.3f} "
                f"right_reward={right_reward_total:.3f} "
                f"score={left_score}-{right_score} goal_diff={left_score - right_score:+d} length={steps}"
            )
    finally:
        left_agent.close()
        right_agent.close()
        env.close()

    summarize("arena_summary", results)


if __name__ == "__main__":
    main()
