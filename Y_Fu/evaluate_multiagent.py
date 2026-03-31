from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from evaluate import evaluate_agent, make_env, resolve_device, set_global_seed
from yfu_football.model import ActorCritic


@dataclass(frozen=True)
class ScenarioSpec:
    label: str
    stage_name: str
    checkpoint_dir: Path
    candidate_names: tuple[str, ...]
    eval_episodes: int
    requested_agent_count: int
    note: str = ""


SCENARIOS: tuple[ScenarioSpec, ...] = (
    ScenarioSpec(
        label="2_agents",
        stage_name="academy_pass_and_shoot_with_keeper",
        checkpoint_dir=Path("Y_Fu/checkpoints/academy_pass_and_shoot_with_keeper"),
        candidate_names=("update_5.pt", "update_20.pt", "update_40.pt"),
        eval_episodes=20,
        requested_agent_count=2,
    ),
    ScenarioSpec(
        label="3_agents",
        stage_name="academy_3_vs_1_with_keeper",
        checkpoint_dir=Path("Y_Fu/checkpoints/academy_3_vs_1_with_keeper"),
        candidate_names=("update_10.pt", "update_90.pt", "latest.pt"),
        eval_episodes=20,
        requested_agent_count=3,
    ),
    ScenarioSpec(
        label="5_agents",
        stage_name="five_vs_five",
        checkpoint_dir=Path("Y_Fu/checkpoints/five_vs_five"),
        candidate_names=("update_10.pt", "update_140.pt", "latest.pt"),
        eval_episodes=5,
        requested_agent_count=5,
        note="Current preset controls 4 players in code, but this folder name follows the requested 5-agent layout.",
    ),
    ScenarioSpec(
        label="11_agents",
        stage_name="full_11v11_residual",
        checkpoint_dir=Path("Y_Fu/checkpoints/full_11v11_residual"),
        candidate_names=("update_20.pt", "update_80.pt", "latest.pt"),
        eval_episodes=5,
        requested_agent_count=11,
    ),
)

VIDEO_SEED_CANDIDATES: tuple[int, ...] = (123, 456, 789, 2026, 7)


class EvalArgs:
    def __init__(self, checkpoint: Path, episodes: int, device: str, deterministic: bool, seed: int | None) -> None:
        self.checkpoint = str(checkpoint)
        self.episodes = episodes
        self.device = device
        self.deterministic = deterministic
        self.render = False
        self.compare_random = False
        self.save_video = False
        self.video_dir = ""
        self.stop_on_success = False
        self.seed = seed
        self.env_name = None
        self.representation = None
        self.rewards = None
        self.num_controlled_players = None
        self.channel_width = None
        self.channel_height = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch-evaluate multi-agent checkpoints and save representative videos.")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--video-root", default="Y_Fu/videos/multiagent")
    parser.add_argument("--report-path", default="Y_Fu/reports/multiagent_eval_report.md")
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--base-seed", type=int, default=123)
    return parser.parse_args()


def load_checkpoint(checkpoint_path: Path, device: torch.device) -> tuple[dict[str, Any], ActorCritic]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    config = checkpoint.get("config", {})
    model = ActorCritic(
        obs_dim=checkpoint["obs_dim"],
        action_dim=checkpoint["action_dim"],
        hidden_sizes=tuple(config.get("hidden_sizes", [256, 256])),
        obs_shape=tuple(checkpoint.get("obs_shape", (checkpoint["obs_dim"],))),
        model_type=config.get("model_type", "auto"),
        feature_dim=int(config.get("feature_dim", 256)),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return config, model


def evaluate_checkpoint(
    checkpoint_path: Path,
    episodes: int,
    device_name: str,
    deterministic: bool,
    seed: int,
) -> tuple[dict[str, float], dict[str, float], dict[str, Any]]:
    device = resolve_device(device_name)
    config, model = load_checkpoint(checkpoint_path, device)
    trained_episode_metrics: list[dict[str, float]] = []
    random_episode_metrics: list[dict[str, float]] = []

    for episode_offset in range(episodes):
        episode_seed = seed + episode_offset
        args = EvalArgs(
            checkpoint=checkpoint_path,
            episodes=1,
            device=device_name,
            deterministic=deterministic,
            seed=episode_seed,
        )

        env = make_env(args, config)
        trained_episode_metrics.append(
            evaluate_agent(
                env=env,
                episodes=1,
                actor=model,
                device=device,
                deterministic=deterministic,
                stop_on_success=False,
            )
        )
        env.close()

        random_env = make_env(args, config)
        random_episode_metrics.append(
            evaluate_agent(
                env=random_env,
                episodes=1,
                actor=None,
                device=device,
                deterministic=deterministic,
                stop_on_success=False,
            )
        )
        random_env.close()

    trained_metrics = aggregate_metrics(trained_episode_metrics)
    random_metrics = aggregate_metrics(random_episode_metrics)
    return trained_metrics, random_metrics, config


def aggregate_metrics(metrics_list: list[dict[str, float]]) -> dict[str, float]:
    aggregated: dict[str, float] = {}
    for key in ("avg_return", "avg_score_reward", "avg_goal_diff", "win_rate", "avg_length"):
        values = [metrics[key] for metrics in metrics_list if math.isfinite(metrics[key])]
        aggregated[key] = float(sum(values) / len(values)) if values else float("nan")
    return aggregated


def choose_representative_checkpoint(results: list[dict[str, Any]]) -> dict[str, Any]:
    def sort_key(item: dict[str, Any]) -> tuple[float, float, float, float]:
        metrics = item["trained_metrics"]
        return (
            metrics["win_rate"],
            metrics["avg_goal_diff"],
            metrics["avg_score_reward"],
            metrics["avg_return"],
        )

    return max(results, key=sort_key)


def find_video_seed(
    checkpoint_path: Path,
    config: dict[str, Any],
    device_name: str,
    deterministic: bool,
) -> tuple[int, dict[str, float]]:
    device = resolve_device(device_name)
    _, model = load_checkpoint(checkpoint_path, device)
    fallback_metrics: dict[str, float] | None = None
    fallback_seed = VIDEO_SEED_CANDIDATES[0]

    for seed in VIDEO_SEED_CANDIDATES:
        args = EvalArgs(
            checkpoint=checkpoint_path,
            episodes=1,
            device=device_name,
            deterministic=deterministic,
            seed=seed,
        )
        env = make_env(args, config)
        metrics = evaluate_agent(
            env=env,
            episodes=1,
            actor=model,
            device=device,
            deterministic=deterministic,
            stop_on_success=False,
        )
        env.close()
        if fallback_metrics is None:
            fallback_metrics = metrics
        if metrics["win_rate"] > 0.0 or metrics["avg_score_reward"] > 0.0:
            return seed, metrics

    assert fallback_metrics is not None
    return fallback_seed, fallback_metrics


def save_video(
    checkpoint_path: Path,
    config: dict[str, Any],
    device_name: str,
    deterministic: bool,
    seed: int,
    video_dir: Path,
) -> None:
    args = EvalArgs(
        checkpoint=checkpoint_path,
        episodes=1,
        device=device_name,
        deterministic=deterministic,
        seed=seed,
    )
    args.save_video = True
    args.video_dir = str(video_dir)

    device = resolve_device(device_name)
    _, model = load_checkpoint(checkpoint_path, device)
    env = make_env(args, config)
    evaluate_agent(
        env=env,
        episodes=1,
        actor=model,
        device=device,
        deterministic=deterministic,
        stop_on_success=False,
    )
    env.close()


def write_report(
    report_path: Path,
    video_root: Path,
    scenario_results: list[dict[str, Any]],
    deterministic: bool,
    base_seed: int,
) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    lines.append("# Multi-Agent Evaluation Report")
    lines.append("")
    lines.append(f"- deterministic_policy: `{deterministic}`")
    lines.append(f"- evaluation_seed_base: `{base_seed}`")
    lines.append(f"- video_root: `{video_root}`")
    lines.append("")

    for scenario in scenario_results:
        spec: ScenarioSpec = scenario["spec"]
        lines.append(f"## {spec.label}")
        lines.append("")
        lines.append(f"- stage: `{spec.stage_name}`")
        lines.append(f"- requested_folder_label: `{spec.requested_agent_count}_agents`")
        if spec.note:
            lines.append(f"- note: {spec.note}")
        lines.append("")
        lines.append("| checkpoint | episodes | avg_score_reward | avg_goal_diff | win_rate | delta_score_vs_random | delta_goal_diff_vs_random | delta_win_rate_vs_random |")
        lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")

        for result in scenario["results"]:
            trained = result["trained_metrics"]
            delta = result["delta_vs_random"]
            lines.append(
                f"| `{result['checkpoint_name']}` | {result['episodes']} | "
                f"{trained['avg_score_reward']:.3f} | {trained['avg_goal_diff']:.3f} | {trained['win_rate']:.3f} | "
                f"{delta['avg_score_reward']:.3f} | {delta['avg_goal_diff']:.3f} | {delta['win_rate']:.3f} |"
            )

        baseline = scenario["results"][0]["trained_metrics"]
        best = scenario["representative"]
        best_metrics = best["trained_metrics"]
        lines.append("")
        lines.append(
            f"- improvement_vs_first_checkpoint: "
            f"`avg_score_reward {best_metrics['avg_score_reward'] - baseline['avg_score_reward']:+.3f}`, "
            f"`avg_goal_diff {best_metrics['avg_goal_diff'] - baseline['avg_goal_diff']:+.3f}`, "
            f"`win_rate {best_metrics['win_rate'] - baseline['win_rate']:+.3f}`"
        )
        lines.append(
            f"- representative_checkpoint: `{best['checkpoint_name']}` "
            f"(video seed `{scenario['video_seed']}`, dir `{video_root / spec.label}`)"
        )
        lines.append("")

    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    set_global_seed(args.base_seed)
    video_root = Path(args.video_root)
    report_path = Path(args.report_path)
    scenario_results: list[dict[str, Any]] = []

    for spec_index, spec in enumerate(SCENARIOS):
        print(f"=== Evaluating {spec.label} ({spec.stage_name}) ===")
        scenario_dir = video_root / spec.label
        scenario_dir.mkdir(parents=True, exist_ok=True)

        results: list[dict[str, Any]] = []
        for checkpoint_name in spec.candidate_names:
            checkpoint_path = spec.checkpoint_dir / checkpoint_name
            if not checkpoint_path.exists():
                print(f"Skipping missing checkpoint: {checkpoint_path}")
                continue

            eval_seed = args.base_seed + spec_index
            print(f"-- checkpoint {checkpoint_name} ({spec.eval_episodes} episodes, seed {eval_seed})")
            trained_metrics, random_metrics, config = evaluate_checkpoint(
                checkpoint_path=checkpoint_path,
                episodes=spec.eval_episodes,
                device_name=args.device,
                deterministic=args.deterministic,
                seed=eval_seed,
            )
            delta_vs_random = {
                key: trained_metrics[key] - random_metrics[key]
                for key in ("avg_return", "avg_score_reward", "avg_goal_diff", "win_rate", "avg_length")
            }
            results.append(
                {
                    "checkpoint_name": checkpoint_name,
                    "checkpoint_path": checkpoint_path,
                    "episodes": spec.eval_episodes,
                    "trained_metrics": trained_metrics,
                    "random_metrics": random_metrics,
                    "delta_vs_random": delta_vs_random,
                    "config": config,
                }
            )

        if not results:
            print(f"No checkpoints evaluated for {spec.label}.")
            continue

        representative = choose_representative_checkpoint(results)
        video_seed, video_seed_metrics = find_video_seed(
            checkpoint_path=representative["checkpoint_path"],
            config=representative["config"],
            device_name=args.device,
            deterministic=args.deterministic,
        )
        print(
            f"-- saving representative video from {representative['checkpoint_name']} "
            f"with seed {video_seed}"
        )
        save_video(
            checkpoint_path=representative["checkpoint_path"],
            config=representative["config"],
            device_name=args.device,
            deterministic=args.deterministic,
            seed=video_seed,
            video_dir=scenario_dir,
        )

        scenario_results.append(
            {
                "spec": spec,
                "results": results,
                "representative": representative,
                "video_seed": video_seed,
                "video_seed_metrics": video_seed_metrics,
            }
        )

    write_report(
        report_path=report_path,
        video_root=video_root,
        scenario_results=scenario_results,
        deterministic=args.deterministic,
        base_seed=args.base_seed,
    )
    print(f"report_written: {report_path}")


if __name__ == "__main__":
    main()
