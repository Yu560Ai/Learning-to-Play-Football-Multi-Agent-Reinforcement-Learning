from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt

def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


project_root = _project_root()
project_root_str = str(project_root)
if project_root_str not in sys.path:
    sys.path.insert(0, project_root_str)

from Two_V_Two.evaluation.run_phase1_eval import evaluate_variant


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a deterministic checkpoint sweep for R3 checkpoints across saved runs.",
    )
    parser.add_argument(
        "--run_root",
        action="append",
        required=True,
        help="Run label and R3 run directory in the form label=path. Repeat for multiple runs.",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=20,
        help="Deterministic evaluation episodes per checkpoint.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="Two_V_Two/results/phase1_extended/r3_checkpoint_sweep",
        help="Directory to write the sweep outputs.",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=10,
        help="Number of top checkpoints to include in the markdown summary.",
    )
    return parser


def parse_run_roots(items: list[str]) -> list[tuple[str, Path]]:
    parsed: list[tuple[str, Path]] = []
    for item in items:
        label, sep, path = item.partition("=")
        if not sep:
            raise ValueError(f"Invalid --run_root value: {item!r}")
        parsed.append((label, Path(path)))
    return parsed


def checkpoint_sort_key(path: Path) -> int:
    stem = path.stem
    if stem.startswith("update_"):
        return int(stem.split("_", 1)[1])
    raise ValueError(f"Unsupported checkpoint name: {path.name}")


def ranking_key(row: dict) -> tuple[float, float, float, float, float]:
    return (
        float(row["mean_pass_to_shot_count"]),
        float(row["mean_pass_count"]),
        float(row["goal_rate"]),
        float(row["mean_goal_count"]),
        -float(row["mean_same_owner_possession_length"]),
    )


def write_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = [
        "run_label",
        "checkpoint_name",
        "checkpoint_update",
        "checkpoint_env_steps",
        "episodes",
        "goal_rate",
        "mean_goal_count",
        "mean_pass_count",
        "mean_pass_to_shot_count",
        "mean_assist_count",
        "mean_same_owner_possession_length",
        "mean_episode_return",
        "mean_episode_length",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name) for name in fieldnames})


def write_markdown(top_rows: list[dict], path: Path) -> None:
    lines = [
        "# R3 Checkpoint Sweep",
        "",
        "| rank | run | checkpoint | env_steps | goal_rate | pass_count | pass_to_shot | assists | possession |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for idx, row in enumerate(top_rows, start=1):
        lines.append(
            "| "
            f"{idx} | "
            f"{row['run_label']} | "
            f"{row.get('checkpoint_name', row.get('checkpoint', ''))} | "
            f"{int(row['checkpoint_env_steps'])} | "
            f"{float(row['goal_rate']):.3f} | "
            f"{float(row['mean_pass_count']):.3f} | "
            f"{float(row['mean_pass_to_shot_count']):.3f} | "
            f"{float(row['mean_assist_count']):.3f} | "
            f"{float(row['mean_same_owner_possession_length']):.2f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_plot(rows: list[dict], path: Path) -> None:
    fig, axes = plt.subplots(5, 1, figsize=(12, 14), sharex=True)
    metrics = [
        ("goal_rate", "Goal Rate"),
        ("mean_pass_count", "Pass Count"),
        ("mean_pass_to_shot_count", "Pass-To-Shot Count"),
        ("mean_assist_count", "Assist Count"),
        ("mean_same_owner_possession_length", "Possession Streak"),
    ]
    labels = sorted({str(row["run_label"]) for row in rows})
    for axis, (metric_key, metric_label) in zip(axes, metrics):
        for label in labels:
            subset = [row for row in rows if row["run_label"] == label]
            subset.sort(key=lambda item: int(item["checkpoint_env_steps"]))
            axis.plot(
                [int(item["checkpoint_env_steps"]) for item in subset],
                [float(item[metric_key]) for item in subset],
                marker="o",
                linewidth=1.6,
                markersize=3.0,
                label=label,
            )
        axis.set_ylabel(metric_label)
        axis.grid(True, alpha=0.25)
    axes[0].legend(loc="best")
    axes[-1].set_xlabel("Environment Steps")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = build_parser().parse_args()
    run_roots = parse_run_roots(args.run_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    for run_label, run_dir in run_roots:
        checkpoint_dir = run_dir / "checkpoints"
        checkpoint_paths = sorted(checkpoint_dir.glob("update_*.pt"), key=checkpoint_sort_key)
        for checkpoint_path in checkpoint_paths:
            summary = evaluate_variant(run_dir, checkpoint_path.name, args.episodes)
            summary["run_label"] = run_label
            summary["checkpoint_name"] = checkpoint_path.name
            rows.append(summary)
            print(
                "[sweep] "
                f"run={run_label} "
            f"checkpoint={checkpoint_path.name} "
                f"env_steps={summary['checkpoint_env_steps']} "
                f"goal_rate={summary['goal_rate']:.3f} "
                f"passes={summary['mean_pass_count']:.3f} "
                f"pass_to_shot={summary['mean_pass_to_shot_count']:.3f} "
                f"assists={summary['mean_assist_count']:.3f} "
                f"possession={summary['mean_same_owner_possession_length']:.2f}",
                flush=True,
            )

    rows.sort(key=lambda item: (str(item["run_label"]), int(item["checkpoint_env_steps"])))
    ranking = sorted(rows, key=ranking_key, reverse=True)

    (output_dir / "sweep_results.json").write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_csv(rows, output_dir / "sweep_results.csv")
    write_markdown(ranking[: args.top_k], output_dir / "top_checkpoints.md")
    write_plot(rows, output_dir / "r3_checkpoint_sweep.png")

    best_row = ranking[0] if ranking else None
    summary = {
        "episodes_per_checkpoint": int(args.episodes),
        "num_checkpoints": len(rows),
        "run_labels": [label for label, _ in run_roots],
        "best_checkpoint": best_row,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"[sweep] output_dir={output_dir}", flush=True)
    if best_row is not None:
        print(
            "[sweep] best "
            f"run={best_row['run_label']} "
            f"checkpoint={best_row.get('checkpoint_name', best_row.get('checkpoint', ''))} "
            f"env_steps={best_row['checkpoint_env_steps']} "
            f"goal_rate={best_row['goal_rate']:.3f} "
            f"passes={best_row['mean_pass_count']:.3f} "
            f"pass_to_shot={best_row['mean_pass_to_shot_count']:.3f} "
            f"assists={best_row['mean_assist_count']:.3f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
