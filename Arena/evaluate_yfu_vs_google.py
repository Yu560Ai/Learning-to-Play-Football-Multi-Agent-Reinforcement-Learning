from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .env import ArenaMatchEnv
from .registry import create_agent


@dataclass
class MatchResult:
    seed: int
    left_name: str
    right_name: str
    left_reward: float
    right_reward: float
    left_score: int
    right_score: int
    length: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Y_Fu 5v5 model against Google built-in baseline.")
    parser.add_argument(
        "--checkpoint",
        default="best_models/Y_Fu/shared_policy_ppo_five_vs_five_Y_Fu_0.0%.pt",
    )
    parser.add_argument("--seed-start", type=int, default=100)
    parser.add_argument("--num-seeds", type=int, default=10)
    parser.add_argument(
        "--report-dir",
        default="Arena/reports/yfu_vs_google_builtin",
    )
    parser.add_argument(
        "--video-dir",
        default="Arena/videos/yfu_vs_google_builtin",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--deterministic-yfu", action="store_true", default=True)
    return parser.parse_args()


def run_single_match(
    *,
    env_seed: int,
    left_agent_name: str,
    right_agent_name: str,
    left_checkpoint: str | None,
    right_checkpoint: str | None,
    deterministic_left: bool,
    deterministic_right: bool,
    device: str,
    save_video: bool = False,
    video_dir: str = "Arena/videos",
) -> MatchResult:
    env = ArenaMatchEnv(
        env_name="5_vs_5",
        representation="extracted",
        action_set="v2",
        left_players=4,
        right_players=4,
        channel_dimensions=(42, 42),
        env_seed=env_seed,
        save_video=save_video,
        video_dir=video_dir,
    )
    left_agent = create_agent(
        left_agent_name,
        checkpoint=left_checkpoint,
        device=device,
        num_actions=env.num_actions,
    )
    right_agent = create_agent(
        right_agent_name,
        checkpoint=right_checkpoint,
        device=device,
        num_actions=env.num_actions,
    )

    left_obs, right_obs = env.reset()
    left_agent.reset()
    right_agent.reset()
    left_reward_total = 0.0
    right_reward_total = 0.0
    steps = 0
    done = False

    try:
        while not done:
            left_action = left_agent.act(left_obs, deterministic=deterministic_left)
            right_action = right_agent.act(right_obs, deterministic=deterministic_right)
            left_obs, right_obs, rewards, done, _ = env.step(left_action, right_action)
            left_reward_total += float(rewards[0])
            right_reward_total += float(rewards[1])
            steps += 1

        left_score, right_score = env.get_score()
        return MatchResult(
            seed=env_seed,
            left_name=left_agent_name,
            right_name=right_agent_name,
            left_reward=left_reward_total,
            right_reward=right_reward_total,
            left_score=left_score,
            right_score=right_score,
            length=steps,
        )
    finally:
        left_agent.close()
        right_agent.close()
        env.close()


def summarize(results: list[MatchResult]) -> dict[str, float]:
    left_rewards = np.asarray([result.left_reward for result in results], dtype=np.float32)
    right_rewards = np.asarray([result.right_reward for result in results], dtype=np.float32)
    left_goals = np.asarray([result.left_score for result in results], dtype=np.float32)
    right_goals = np.asarray([result.right_score for result in results], dtype=np.float32)
    lengths = np.asarray([result.length for result in results], dtype=np.float32)
    left_wins = np.asarray([1.0 if result.left_score > result.right_score else 0.0 for result in results], dtype=np.float32)
    draws = np.asarray([1.0 if result.left_score == result.right_score else 0.0 for result in results], dtype=np.float32)
    return {
        "left_avg_reward": float(left_rewards.mean()),
        "right_avg_reward": float(right_rewards.mean()),
        "left_avg_goals": float(left_goals.mean()),
        "right_avg_goals": float(right_goals.mean()),
        "avg_goal_diff": float((left_goals - right_goals).mean()),
        "left_win_rate": float(left_wins.mean()),
        "draw_rate": float(draws.mean()),
        "avg_length": float(lengths.mean()),
    }


def format_summary(summary: dict[str, float]) -> str:
    return (
        f"left_avg_reward={summary['left_avg_reward']:.3f} "
        f"right_avg_reward={summary['right_avg_reward']:.3f} "
        f"avg_score={summary['left_avg_goals']:.3f}-{summary['right_avg_goals']:.3f} "
        f"avg_goal_diff={summary['avg_goal_diff']:.3f} "
        f"left_win_rate={summary['left_win_rate']:.3f} "
        f"draw_rate={summary['draw_rate']:.3f} "
        f"avg_length={summary['avg_length']:.1f}"
    )


def write_report(
    report_path: Path,
    checkpoint: str,
    representative_seed: int,
    left_results: list[MatchResult],
    right_results: list[MatchResult],
    left_summary: dict[str, float],
    right_summary: dict[str, float],
    video_dir: Path,
) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    lines.append("Y_Fu vs Google Built-In Arena Evaluation")
    lines.append("")
    lines.append(f"checkpoint: {checkpoint}")
    lines.append("environment: 5_vs_5")
    lines.append("representation: extracted")
    lines.append("action_set: v2")
    lines.append("controlled_players_per_side: 4")
    lines.append("")
    lines.append("Direction A: Y_Fu on left vs Google built-in on right")
    lines.append(f"summary: {format_summary(left_summary)}")
    lines.append("per-seed results:")
    for result in left_results:
        lines.append(
            f"seed={result.seed} score={result.left_score}-{result.right_score} "
            f"goal_diff={result.left_score - result.right_score:+d} "
            f"left_reward={result.left_reward:.3f} right_reward={result.right_reward:.3f} "
            f"length={result.length}"
        )
    lines.append("")
    lines.append("Direction B: Google built-in on left vs Y_Fu on right")
    lines.append(f"summary: {format_summary(right_summary)}")
    lines.append("per-seed results:")
    for result in right_results:
        lines.append(
            f"seed={result.seed} score={result.left_score}-{result.right_score} "
            f"goal_diff={result.left_score - result.right_score:+d} "
            f"left_reward={result.left_reward:.3f} right_reward={result.right_reward:.3f} "
            f"length={result.length}"
        )
    lines.append("")
    lines.append("Combined interpretation:")
    yfu_wins = sum(1 for result in left_results if result.left_score > result.right_score) + sum(
        1 for result in right_results if result.right_score > result.left_score
    )
    google_wins = sum(1 for result in left_results if result.right_score > result.left_score) + sum(
        1 for result in right_results if result.left_score > result.right_score
    )
    draws = sum(1 for result in left_results if result.left_score == result.right_score) + sum(
        1 for result in right_results if result.left_score == result.right_score
    )
    lines.append(f"combined_record: Y_Fu wins={yfu_wins}, Google wins={google_wins}, draws={draws}")
    lines.append(f"representative_video_seed: {representative_seed}")
    lines.append(f"representative_video_dir: {video_dir}")
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    seeds = [args.seed_start + offset for offset in range(args.num_seeds)]
    report_dir = Path(args.report_dir)
    video_dir = Path(args.video_dir)

    left_results: list[MatchResult] = []
    right_results: list[MatchResult] = []

    for seed in seeds:
        print(f"[run] seed={seed} direction=Y_Fu_left")
        left_results.append(
            run_single_match(
                env_seed=seed,
                left_agent_name="yfu_multiagent",
                right_agent_name="google_builtin",
                left_checkpoint=args.checkpoint,
                right_checkpoint=None,
                deterministic_left=args.deterministic_yfu,
                deterministic_right=False,
                device=args.device,
            )
        )

    for seed in seeds:
        print(f"[run] seed={seed} direction=Y_Fu_right")
        right_results.append(
            run_single_match(
                env_seed=seed,
                left_agent_name="google_builtin",
                right_agent_name="yfu_multiagent",
                left_checkpoint=None,
                right_checkpoint=args.checkpoint,
                deterministic_left=False,
                deterministic_right=args.deterministic_yfu,
                device=args.device,
            )
        )

    left_summary = summarize(left_results)
    right_summary = summarize(right_results)

    representative_seed = seeds[0]
    print(f"[video] seed={representative_seed}")
    run_single_match(
        env_seed=representative_seed,
        left_agent_name="yfu_multiagent",
        right_agent_name="google_builtin",
        left_checkpoint=args.checkpoint,
        right_checkpoint=None,
        deterministic_left=args.deterministic_yfu,
        deterministic_right=False,
        device=args.device,
        save_video=True,
        video_dir=str(video_dir),
    )

    report_path = report_dir / "yfu_vs_google_builtin_5v5_10seeds.txt"
    write_report(
        report_path=report_path,
        checkpoint=args.checkpoint,
        representative_seed=representative_seed,
        left_results=left_results,
        right_results=right_results,
        left_summary=left_summary,
        right_summary=right_summary,
        video_dir=video_dir,
    )
    print(f"report_written: {report_path}")


if __name__ == "__main__":
    main()
