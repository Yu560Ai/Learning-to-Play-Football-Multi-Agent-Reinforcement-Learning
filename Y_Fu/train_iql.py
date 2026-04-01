from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from evaluate import evaluate_agent, make_env, print_summary, resolve_device, set_global_seed
from yfu_football.iql import DiscreteIQL, IQLConfig
from yfu_football.offline_dataset import OfflineGRFDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train discrete IQL on offline GRF data.")
    parser.add_argument("--dataset-dirs", nargs="+", required=True)
    parser.add_argument("--reward-key", choices=("reward", "score_reward"), default="reward")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--save-dir", default="Y_Fu/checkpoints/iql")
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.993)
    parser.add_argument("--expectile", type=float, default=0.7)
    parser.add_argument("--temperature", type=float, default=3.0)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--total-gradient-steps", type=int, default=1_000_000)
    parser.add_argument("--eval-interval", type=int, default=10_000)
    parser.add_argument("--save-interval", type=int, default=50_000)
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--chunk-reuse-batches", type=int, default=32)
    parser.add_argument("--reward-norm-eps", type=float, default=1e-6)
    parser.add_argument("--normalize-rewards", dest="normalize_rewards", action="store_true")
    parser.add_argument("--no-normalize-rewards", dest="normalize_rewards", action="store_false")
    parser.add_argument("--seed", type=int, default=42)
    parser.set_defaults(normalize_rewards=True)
    return parser.parse_args()


def _compare_eval_metrics(metrics: dict[str, float]) -> tuple[float, float, float]:
    return (
        float(metrics["win_rate"]),
        float(metrics["avg_goal_diff"]),
        float(metrics["avg_score_reward"]),
    )


def main() -> None:
    args = parse_args()
    set_global_seed(args.seed)
    device = resolve_device(args.device)

    dataset = OfflineGRFDataset(
        args.dataset_dirs,
        reward_key=args.reward_key,
        chunk_reuse_batches=args.chunk_reuse_batches,
    )
    stats = dataset.get_stats()
    reward_norm_scale = 1.0
    reward_norm_source = "disabled"
    if args.normalize_rewards:
        episode_return_std = float(stats["episode_return_std"])
        reward_std = float(stats["reward_std"])
        if episode_return_std > args.reward_norm_eps:
            reward_norm_scale = episode_return_std
            reward_norm_source = "episode_return_std"
        elif reward_std > args.reward_norm_eps:
            reward_norm_scale = reward_std
            reward_norm_source = "reward_std_fallback"
        else:
            reward_norm_scale = 1.0
            reward_norm_source = "fallback_identity"
    print(
        "dataset_stats: "
        f"num_chunks={stats['num_chunks']} "
        f"num_transitions={stats['num_transitions']} "
        f"num_episodes={stats['num_episodes']} "
        f"num_player_episodes={stats['num_player_episodes']} "
        f"reward_mean={stats['reward_mean']:.5f} "
        f"reward_std={stats['reward_std']:.5f} "
        f"episode_return_mean={stats['episode_return_mean']:.5f} "
        f"episode_return_std={stats['episode_return_std']:.5f} "
        f"chunk_reuse_batches={stats['chunk_reuse_batches']} "
        f"reward_norm_enabled={args.normalize_rewards} "
        f"reward_norm_scale={reward_norm_scale:.6f} "
        f"reward_norm_source={reward_norm_source}"
    )

    config = IQLConfig(
        obs_dim=dataset.obs_dim,
        action_dim=dataset.action_dim,
        obs_shape=dataset.obs_shape,
        feature_dim=256,
        hidden_sizes=(256, 256),
        encoder_type="cnn",
        env_name=str(dataset.env_config["env_name"]),
        representation=str(dataset.env_config["representation"]),
        rewards=str(dataset.env_config["rewards"]),
        num_controlled_players=int(dataset.env_config["num_controlled_players"]),
        channel_dimensions=tuple(int(dim) for dim in dataset.env_config["channel_dimensions"]),
        pass_success_reward=float(dataset.env_config.get("reward_shaping", {}).get("pass_success_reward", 0.0)),
        pass_failure_penalty=float(dataset.env_config.get("reward_shaping", {}).get("pass_failure_penalty", 0.0)),
        pass_progress_reward_scale=float(
            dataset.env_config.get("reward_shaping", {}).get("pass_progress_reward_scale", 0.0)
        ),
        shot_attempt_reward=float(dataset.env_config.get("reward_shaping", {}).get("shot_attempt_reward", 0.0)),
        attacking_possession_reward=float(
            dataset.env_config.get("reward_shaping", {}).get("attacking_possession_reward", 0.0)
        ),
        attacking_x_threshold=float(dataset.env_config.get("reward_shaping", {}).get("attacking_x_threshold", 0.55)),
        final_third_entry_reward=float(
            dataset.env_config.get("reward_shaping", {}).get("final_third_entry_reward", 0.0)
        ),
        possession_retention_reward=float(
            dataset.env_config.get("reward_shaping", {}).get("possession_retention_reward", 0.0)
        ),
        possession_recovery_reward=float(
            dataset.env_config.get("reward_shaping", {}).get("possession_recovery_reward", 0.0)
        ),
        defensive_third_recovery_reward=float(
            dataset.env_config.get("reward_shaping", {}).get("defensive_third_recovery_reward", 0.0)
        ),
        opponent_attacking_possession_penalty=float(
            dataset.env_config.get("reward_shaping", {}).get("opponent_attacking_possession_penalty", 0.0)
        ),
        own_half_turnover_penalty=float(
            dataset.env_config.get("reward_shaping", {}).get("own_half_turnover_penalty", 0.0)
        ),
        own_half_x_threshold=float(dataset.env_config.get("reward_shaping", {}).get("own_half_x_threshold", 0.0)),
        defensive_x_threshold=float(dataset.env_config.get("reward_shaping", {}).get("defensive_x_threshold", -0.45)),
        pending_pass_horizon=int(dataset.env_config.get("reward_shaping", {}).get("pending_pass_horizon", 8)),
        learning_rate=args.learning_rate,
        gamma=args.gamma,
        expectile=args.expectile,
        temperature=args.temperature,
        tau=args.tau,
        batch_size=args.batch_size,
        total_gradient_steps=args.total_gradient_steps,
        eval_interval=args.eval_interval,
        reward_normalization_enabled=bool(args.normalize_rewards),
        reward_normalization_scale=float(reward_norm_scale),
        reward_normalization_source=reward_norm_source,
        seed=args.seed,
    )
    iql = DiscreteIQL(config=config, device=device)

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    training_metadata = {
        "dataset_dirs": [str(path) for path in args.dataset_dirs],
        "reward_key": args.reward_key,
        "chunk_reuse_batches": args.chunk_reuse_batches,
        "normalize_rewards": bool(args.normalize_rewards),
        "reward_norm_scale": float(reward_norm_scale),
        "reward_norm_source": reward_norm_source,
        "dataset_stats": stats,
        "iql_config": asdict(config),
    }
    (save_dir / "training_metadata.json").write_text(
        json.dumps(training_metadata, indent=2),
        encoding="utf-8",
    )

    best_metrics = {
        "win_rate": float("-inf"),
        "avg_goal_diff": float("-inf"),
        "avg_score_reward": float("-inf"),
    }
    best_checkpoint_path = save_dir / "best.pt"

    for gradient_step in range(1, args.total_gradient_steps + 1):
        batch = dataset.sample_batch(args.batch_size, device=device)
        if args.normalize_rewards and reward_norm_scale > 0.0:
            batch["reward"] = batch["reward"] / float(reward_norm_scale)
        update_metrics = iql.update(batch)

        if gradient_step % 100 == 0:
            print(
                f"[train {gradient_step}/{args.total_gradient_steps}] "
                f"q_loss={update_metrics['q_loss']:.6f} "
                f"q1_loss={update_metrics['q1_loss']:.6f} "
                f"q2_loss={update_metrics['q2_loss']:.6f} "
                f"v_loss={update_metrics['v_loss']:.6f} "
                f"mean_q={update_metrics['mean_q']:.4f} "
                f"mean_v={update_metrics['mean_v']:.4f} "
                f"mean_target_q={update_metrics['mean_target_q']:.4f}"
            )

        if gradient_step % args.eval_interval == 0:
            eval_env = make_env(
                argparse.Namespace(
                    checkpoint="",
                    episodes=args.eval_episodes,
                    device=args.device,
                    deterministic=True,
                    render=False,
                    compare_random=False,
                    save_video=False,
                    video_dir="",
                    stop_on_success=False,
                    seed=args.seed,
                    env_name=config.env_name,
                    representation=config.representation,
                    rewards=config.rewards,
                    num_controlled_players=config.num_controlled_players,
                    channel_width=config.channel_dimensions[0],
                    channel_height=config.channel_dimensions[1],
                ),
                asdict(config),
            )
            eval_metrics = evaluate_agent(
                env=eval_env,
                episodes=args.eval_episodes,
                actor=iql,
                device=device,
                deterministic=True,
                stop_on_success=False,
            )
            eval_env.close()
            print_summary("iql_eval", eval_metrics)

            if _compare_eval_metrics(eval_metrics) > _compare_eval_metrics(best_metrics):
                best_metrics = eval_metrics
                iql.save_checkpoint(best_checkpoint_path)
                print(f"best_checkpoint: {best_checkpoint_path}")

        if gradient_step % args.save_interval == 0:
            iql.save_checkpoint(save_dir / f"step_{gradient_step}.pt")

    latest_checkpoint = iql.save_checkpoint(save_dir / "latest.pt")
    print(f"Training finished. Latest checkpoint: {latest_checkpoint}")
    if best_checkpoint_path.exists():
        print(f"Best checkpoint: {best_checkpoint_path}")


if __name__ == "__main__":
    main()
