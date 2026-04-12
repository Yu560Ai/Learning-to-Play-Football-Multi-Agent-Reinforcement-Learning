from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import numpy as np


METRICS = [
    ("mean_episode_return", "Episode Return"),
    ("mean_goal_count", "Goal Count"),
    ("mean_pass_count", "Pass Count"),
    ("mean_pass_to_shot_count", "Pass-To-Shot Count"),
    ("mean_assist_count", "Assist Count"),
    ("mean_same_owner_possession_length", "Same-Owner Possession Length"),
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Refresh aggregated monitoring curves across seed runs for one chosen condition.",
    )
    parser.add_argument("--run_root", required=True, help="Directory containing seed_*/ run directories.")
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[1, 2, 3],
        help="Seed ids to include, matching run_root/seed_<id>.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory for aggregated plots. Defaults under Two_V_Two/results/monitoring/<group>/curves.",
    )
    return parser


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def load_seed_run(run_root: Path, seed: int) -> tuple[Path, dict[str, Any], list[dict[str, Any]]]:
    run_dir = run_root / f"seed_{seed}"
    metrics_path = run_dir / "metrics.jsonl"
    config_path = run_dir / "config.json"
    if not run_dir.exists():
        raise FileNotFoundError(f"Missing run directory for seed {seed}: {run_dir}")
    if not metrics_path.exists():
        raise FileNotFoundError(f"Missing metrics.jsonl for seed {seed}: {metrics_path}")
    if not config_path.exists():
        raise FileNotFoundError(f"Missing config.json for seed {seed}: {config_path}")
    config = json.loads(config_path.read_text())
    rows = load_jsonl(metrics_path)
    if not rows:
        raise ValueError(f"metrics.jsonl is empty for seed {seed}: {metrics_path}")
    return run_dir, config, rows


def infer_output_dir(run_root: Path) -> Path:
    return Path("Two_V_Two/results/monitoring") / run_root.name / "curves"


def is_number(value: Any) -> bool:
    if value is None:
        return False
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return not math.isnan(number)


def summarize_group(configs: list[dict[str, Any]]) -> dict[str, Any]:
    scenario_names = sorted({str(cfg.get("scenario_name", "")) for cfg in configs})
    reward_variants = sorted({str(cfg.get("reward_variant", "")) for cfg in configs})
    structure_variants = sorted({str(cfg.get("structure_variant", "")) for cfg in configs})
    return {
        "scenario_names": scenario_names,
        "reward_variants": reward_variants,
        "structure_variants": structure_variants,
    }


def aggregate_metric(rows_by_seed: dict[int, list[dict[str, Any]]], metric_key: str) -> dict[str, list[float]]:
    env_steps = sorted({int(row["env_steps"]) for rows in rows_by_seed.values() for row in rows})
    mean_values: list[float] = []
    std_values: list[float] = []
    min_values: list[float] = []
    max_values: list[float] = []
    counts: list[int] = []

    per_seed_lookup = {
        seed: {int(row["env_steps"]): row for row in rows}
        for seed, rows in rows_by_seed.items()
    }

    for env_step in env_steps:
        values = []
        for lookup in per_seed_lookup.values():
            row = lookup.get(env_step)
            if row is None:
                continue
            value = row.get(metric_key)
            if is_number(value):
                values.append(float(value))
        if values:
            arr = np.asarray(values, dtype=np.float32)
            mean_values.append(float(arr.mean()))
            std_values.append(float(arr.std()))
            min_values.append(float(arr.min()))
            max_values.append(float(arr.max()))
            counts.append(int(arr.size))
        else:
            mean_values.append(float("nan"))
            std_values.append(float("nan"))
            min_values.append(float("nan"))
            max_values.append(float("nan"))
            counts.append(0)

    return {
        "env_steps": env_steps,
        "mean": mean_values,
        "std": std_values,
        "min": min_values,
        "max": max_values,
        "count": counts,
    }


def main() -> None:
    args = build_parser().parse_args()
    run_root = Path(args.run_root)
    output_dir = Path(args.output_dir) if args.output_dir else infer_output_dir(run_root)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows_by_seed: dict[int, list[dict[str, Any]]] = {}
    configs: list[dict[str, Any]] = []
    seed_run_dirs: dict[int, str] = {}

    for seed in args.seeds:
        run_dir, config, rows = load_seed_run(run_root, seed)
        rows_by_seed[int(seed)] = rows
        configs.append(config)
        seed_run_dirs[int(seed)] = str(run_dir)

    group_summary = summarize_group(configs)
    if group_summary["scenario_names"] != ["two_v_two_plus_goalkeepers"]:
        raise ValueError(
            "Monitoring workflow is restricted to the corrected custom scenario only. "
            f"Found scenarios={group_summary['scenario_names']}"
        )

    aggregates = {metric_key: aggregate_metric(rows_by_seed, metric_key) for metric_key, _ in METRICS}

    fig, axes = plt.subplots(len(METRICS), 1, figsize=(12, 18), sharex=True)
    seed_colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd", "#8c564b"]

    for axis, (metric_key, metric_label) in zip(axes, METRICS):
        for color, seed in zip(seed_colors, sorted(rows_by_seed)):
            rows = rows_by_seed[seed]
            xs = [int(row["env_steps"]) for row in rows]
            ys = [float(row.get(metric_key, float("nan"))) for row in rows]
            axis.plot(xs, ys, color=color, alpha=0.22, linewidth=1.0)

        xs = aggregates[metric_key]["env_steps"]
        mean = np.asarray(aggregates[metric_key]["mean"], dtype=np.float32)
        std = np.asarray(aggregates[metric_key]["std"], dtype=np.float32)
        valid = ~np.isnan(mean)
        if np.any(valid):
            xs_valid = np.asarray(xs, dtype=np.int64)[valid]
            mean_valid = mean[valid]
            std_valid = std[valid]
            axis.plot(xs_valid, mean_valid, color="#111111", linewidth=2.0, label="seed mean")
            axis.fill_between(
                xs_valid,
                mean_valid - std_valid,
                mean_valid + std_valid,
                color="#666666",
                alpha=0.18,
                label="seed std" if metric_key == METRICS[0][0] else None,
            )
        axis.set_ylabel(metric_label)
        axis.grid(True, alpha=0.25)

    axes[0].legend(loc="best")
    axes[-1].set_xlabel("Environment Steps")
    title = (
        f"Monitoring Curves | reward={','.join(group_summary['reward_variants'])} "
        f"| structure={','.join(group_summary['structure_variants'])}"
    )
    fig.suptitle(title, fontsize=14)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.985))

    figure_path = output_dir / "learning_curves.png"
    fig.savefig(figure_path, dpi=180)
    plt.close(fig)

    summary = {
        "run_root": str(run_root),
        "seed_run_dirs": seed_run_dirs,
        "group": group_summary,
        "metrics": {
            metric_key: aggregates[metric_key]
            for metric_key, _ in METRICS
        },
        "figure": str(figure_path),
    }
    (output_dir / "curve_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"[monitor_curves] figure={figure_path}", flush=True)


if __name__ == "__main__":
    main()
