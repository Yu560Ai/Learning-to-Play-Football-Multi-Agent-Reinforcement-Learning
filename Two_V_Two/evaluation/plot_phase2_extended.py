from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt


CONDITIONS = [
    ("r2_progress", "shared_ppo"),
    ("r2_progress", "mappo_id_cc"),
    ("r3_assist", "shared_ppo"),
    ("r3_assist", "mappo_id_cc"),
]

METRICS = [
    ("mean_episode_return", "Episode Return"),
    ("mean_pass_count", "Pass Count"),
    ("mean_pass_to_shot_count", "Pass-To-Shot Count"),
    ("mean_assist_count", "Assist Count"),
    ("mean_same_owner_possession_length", "Possession Length"),
]


def condition_label(reward_variant: str, structure_variant: str) -> str:
    return f"{reward_variant}/{structure_variant}"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plot focused Phase 2 training curves from metrics.jsonl files.",
    )
    parser.add_argument(
        "--results_root",
        type=str,
        default="Two_V_Two/results/phase2_extended",
        help="Root directory containing per-reward and per-structure metrics.jsonl files.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="Two_V_Two/results/phase2_extended/analysis",
        help="Directory to write the training-curve plots.",
    )
    return parser


def load_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main() -> None:
    args = build_parser().parse_args()
    results_root = Path(args.results_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data: dict[str, list[dict]] = {}
    for reward_variant, structure_variant in CONDITIONS:
        metrics_path = results_root / reward_variant / structure_variant / "metrics.jsonl"
        if not metrics_path.exists():
            continue
        data[condition_label(reward_variant, structure_variant)] = load_rows(metrics_path)

    fig, axes = plt.subplots(len(METRICS), 1, figsize=(12, 16), sharex=True)
    for axis, (metric_key, metric_label) in zip(axes, METRICS):
        for label, rows in sorted(data.items()):
            axis.plot(
                [int(row["env_steps"]) for row in rows],
                [float(row.get(metric_key, 0.0)) for row in rows],
                linewidth=1.8,
                label=label,
            )
        axis.set_ylabel(metric_label)
        axis.grid(True, alpha=0.25)
    axes[0].legend(loc="best")
    axes[-1].set_xlabel("Environment Steps")
    fig.tight_layout()
    figure_path = output_dir / "training_curves.png"
    fig.savefig(figure_path, dpi=180)
    plt.close(fig)

    summary = {
        "conditions": sorted(data.keys()),
        "metrics": [metric_key for metric_key, _ in METRICS],
        "figure": str(figure_path),
    }
    (output_dir / "training_curve_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"[phase2_plot] figure={figure_path}", flush=True)


if __name__ == "__main__":
    main()
