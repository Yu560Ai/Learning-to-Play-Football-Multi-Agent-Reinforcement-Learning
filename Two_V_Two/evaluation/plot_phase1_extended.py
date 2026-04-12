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

PLOT_SPECS = [
    ("mean_episode_return", "Episode Return", "plot_0_return_curve.png"),
    ("mean_goal_count", "Goal Count Proxy", "plot_a_goal_curve.png"),
    ("mean_pass_count", "Pass Count", "plot_b_pass_curve.png"),
    ("mean_assist_count", "Assist Count", "plot_c_assist_curve.png"),
    ("mean_same_owner_possession_length", "Mean Same-Agent Possession Streak", "plot_d_possession_curve.png"),
    ("mean_pass_to_shot_count", "Pass-To-Shot Count", "plot_f_pass_to_shot_curve.png"),
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate Phase 1 extended plots across budgets and reward variants.",
    )
    parser.add_argument(
        "--budget_root",
        action="append",
        default=[],
        help="Budget label and result root in the form label=path. Repeat for multiple budgets.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="Two_V_Two/results/phase1_extended/plots",
        help="Directory where plots and summaries will be written.",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=5,
        help="Moving-average window for training-curve smoothing.",
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
    if array.size == 0 or window <= 1 or array.size < window:
        return array
    kernel = np.ones(window, dtype=np.float64) / float(window)
    smoothed = np.convolve(array, kernel, mode="valid")
    prefix = array[: window - 1]
    return np.concatenate([prefix, smoothed])


def parse_budget_roots(items: list[str]) -> list[tuple[str, Path]]:
    parsed: list[tuple[str, Path]] = []
    for item in items:
        label, sep, path = item.partition("=")
        if not sep:
            raise ValueError(f"Invalid --budget_root value: {item!r}")
        parsed.append((label, Path(path)))
    return parsed


def last_finite(values: list[float]) -> float:
    array = np.asarray(values, dtype=np.float64)
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        return float("nan")
    return float(finite[-1])


def main() -> None:
    args = build_parser().parse_args()
    budget_roots = parse_budget_roots(args.budget_root)
    if not budget_roots:
        raise ValueError("Provide at least one --budget_root label=path.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary: dict[str, dict[str, dict[str, float]]] = {}

    for metric_key, metric_label, filename in PLOT_SPECS:
        fig, axes = plt.subplots(len(budget_roots), 1, figsize=(11, 4 * len(budget_roots)), sharex=False)
        if len(budget_roots) == 1:
            axes = [axes]

        for axis, (budget_label, root) in zip(axes, budget_roots):
            summary.setdefault(budget_label, {})
            for variant in VARIANTS:
                metrics_path = root / variant / "metrics.jsonl"
                if not metrics_path.exists():
                    continue
                rows = load_metrics(metrics_path)
                steps = [float(row["env_steps"]) for row in rows]
                values = [float(row.get(metric_key, float("nan"))) for row in rows]
                axis.plot(steps, moving_average(values, args.window), linewidth=2, label=variant)
                summary[budget_label].setdefault(variant, {})
                summary[budget_label][variant][metric_key] = last_finite(values)

            axis.set_title(f"{metric_label} - {budget_label}")
            axis.set_xlabel("Environment Steps")
            axis.set_ylabel(metric_label)
            axis.grid(True, alpha=0.25)
            axis.legend(loc="best")

        fig.tight_layout()
        fig.savefig(output_dir / filename, dpi=180)
        plt.close(fig)

    # Plot E: budget comparison summary using final available values.
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    comparison_metrics = [
        ("mean_episode_return", "Episode Return"),
        ("mean_goal_count", "Goal Count Proxy"),
        ("mean_pass_count", "Pass Count"),
        ("mean_assist_count", "Assist Count"),
        ("mean_same_owner_possession_length", "Possession Streak"),
        ("mean_pass_to_shot_count", "Pass-To-Shot Count"),
    ]
    for axis, (metric_key, title) in zip(axes.flatten(), comparison_metrics):
        x = np.arange(len(budget_roots))
        width = 0.18
        for idx, variant in enumerate(VARIANTS):
            values = []
            for budget_label, _root in budget_roots:
                values.append(summary.get(budget_label, {}).get(variant, {}).get(metric_key, float("nan")))
            axis.bar(x + (idx - 1.5) * width, values, width=width, label=variant)
        axis.set_title(title)
        axis.set_xticks(x)
        axis.set_xticklabels([label for label, _ in budget_roots])
        axis.grid(True, axis="y", alpha=0.25)
    axes[0, 0].legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_dir / "plot_e_budget_comparison.png", dpi=180)
    plt.close(fig)

    summary_path = output_dir / "extended_plot_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"[plot] output_dir={output_dir}", flush=True)
    print(f"[plot] summary_json={summary_path}", flush=True)


if __name__ == "__main__":
    main()
