from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

from evaluate import evaluate_agent, make_env, parse_args, print_summary, resolve_device, set_global_seed
from yfu_football.iql import DiscreteIQL


def main() -> None:
    args = parse_args()
    set_global_seed(args.seed)
    checkpoint_path = Path(args.checkpoint)
    iql = DiscreteIQL.load_checkpoint(checkpoint_path, device=resolve_device(args.device))
    config = asdict(iql.config)

    env = make_env(args, config)
    device = resolve_device(args.device)
    trained_metrics = evaluate_agent(
        env=env,
        episodes=args.episodes,
        actor=iql,
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
