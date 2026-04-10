from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


VARIANTS = [
    "r1_scoring",
    "r2_progress",
    "r3_assist",
    "r4_anti_selfish",
]

CURVES = [
    ("mean_episode_return", "Mean Episode Return"),
    ("mean_goal_count", "Mean Goal Count"),
    ("mean_pass_count", "Mean Pass Count"),
    ("mean_assist_count", "Mean Assist Count"),
    ("mean_same_owner_possession_length", "Mean Same-Owner Possession Length"),
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plot Phase 1 training curves across reward variants.",
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=VARIANTS,
        default=VARIANTS,
        help="Reward variants to plot.",
    )
    parser.add_argument(
        "--results_root",
        type=str,
        default="Two_V_Two/results/phase1",
        help="Root directory containing per-variant metrics.jsonl files.",
    )
    parser.add_argument(
        "--output_png",
        type=str,
        default="Two_V_Two/results/phase1/phase1_curves.png",
        help="Path to save the combined curve plot.",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=3,
        help="Moving-average window for smoothing each metric curve.",
    )
    parser.add_argument(
        "--output_json",
        type=str,
        default="Two_V_Two/results/phase1/curve_summary.json",
        help="Path to save the last-20-update summary JSON.",
    )
    return parser


def load_metrics(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def moving_average(values: list[float], window: int) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return array
    if window <= 1 or array.size < window:
        return array
    kernel = np.ones(window, dtype=np.float64) / float(window)
    smoothed = np.convolve(array, kernel, mode="valid")
    prefix = array[: window - 1]
    return np.concatenate([prefix, smoothed])


def finite_tail_mean(values: list[float], count: int = 20) -> float:
    tail = np.asarray(values[-count:], dtype=np.float64)
    tail = tail[np.isfinite(tail)]
    if tail.size == 0:
        return float("nan")
    return float(np.mean(tail))


def main() -> None:
    args = build_parser().parse_args()
    results_root = Path(args.results_root)
    output_png = Path(args.output_png)
    output_png.parent.mkdir(parents=True, exist_ok=True)

    variant_rows: dict[str, list[dict]] = {}
    for variant in args.variants:
        metrics_path = results_root / variant / "metrics.jsonl"
        variant_rows[variant] = load_metrics(metrics_path)

    fig, axes = plt.subplots(len(CURVES), 1, figsize=(11, 18), sharex=True)
    summary: dict[str, dict[str, float]] = {}

    for variant, rows in variant_rows.items():
        env_steps = [float(row["env_steps"]) for row in rows]
        summary[variant] = {}
        for axis, (metric_key, label) in zip(axes, CURVES):
            metric_values = [float(row.get(metric_key, float("nan"))) for row in rows]
            smoothed = moving_average(metric_values, args.window)
            axis.plot(env_steps, smoothed, label=variant, linewidth=2)
            axis.set_ylabel(label)
            axis.grid(True, alpha=0.25)
            summary[variant][metric_key] = finite_tail_mean(metric_values)

    axes[0].set_title("Phase 1 Training Curves Across Reward Variants")
    axes[-1].set_xlabel("Environment Steps")
    axes[0].legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_png, dpi=180)
    plt.close(fig)

    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"[plot] curves_png={output_png}", flush=True)
    print(f"[plot] curve_summary_json={output_json}", flush=True)


if __name__ == "__main__":
    main()
