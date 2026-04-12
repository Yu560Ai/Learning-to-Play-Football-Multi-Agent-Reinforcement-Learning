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


CONDITIONS = [
    ("r2_progress", "shared_ppo"),
    ("r2_progress", "mappo_id_cc"),
    ("r3_assist", "shared_ppo"),
    ("r3_assist", "mappo_id_cc"),
]


def condition_label(reward_variant: str, structure_variant: str) -> str:
    return f"{reward_variant}/{structure_variant}"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run deterministic checkpoint evaluation across the focused Phase 2 extension conditions.",
    )
    parser.add_argument(
        "--results_root",
        type=str,
        default="Two_V_Two/results/phase2_extended",
        help="Root directory containing per-reward and per-structure run directories.",
    )
    parser.add_argument(
        "--conditions",
        nargs="+",
        choices=[condition_label(*item) for item in CONDITIONS],
        default=[condition_label(*item) for item in CONDITIONS],
        help="Subset of Phase 2 extension conditions to evaluate.",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=20,
        help="Deterministic evaluation episodes per checkpoint.",
    )
    parser.add_argument(
        "--checkpoint_stride",
        type=int,
        default=1,
        help="Evaluate every Nth saved checkpoint per run.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="Two_V_Two/results/phase2_extended/analysis",
        help="Directory to write checkpoint evaluation outputs.",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=12,
        help="Number of top checkpoints to include in the markdown summary.",
    )
    return parser


def checkpoint_sort_key(path: Path) -> int:
    stem = path.stem
    if stem.startswith("update_"):
        return int(stem.split("_", 1)[1])
    raise ValueError(f"Unsupported checkpoint name: {path.name}")


def ranking_key(row: dict[str, float | int | str]) -> tuple[float, float, float, float, float, float]:
    return (
        float(row["mean_pass_to_shot_count"]),
        float(row["mean_pass_count"]),
        float(row["goal_rate"]),
        float(row["mean_goal_count"]),
        float(row["mean_assist_count"]),
        -float(row["mean_same_owner_possession_length"]),
    )


def goal_key(row: dict[str, float | int | str]) -> tuple[float, float, float, float]:
    return (
        float(row["goal_rate"]),
        float(row["mean_goal_count"]),
        float(row["mean_pass_to_shot_count"]),
        float(row["mean_pass_count"]),
    )


def quality_score(row: dict[str, float | int | str]) -> float:
    return (
        1000.0 * float(row["mean_pass_to_shot_count"])
        + 100.0 * float(row["mean_pass_count"])
        + 20.0 * float(row["goal_rate"])
        + 5.0 * float(row["mean_goal_count"])
        + 5.0 * float(row["mean_assist_count"])
        - 0.1 * float(row["mean_same_owner_possession_length"])
    )


def write_csv(rows: list[dict[str, float | int | str]], path: Path) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = [
        "condition",
        "reward_variant",
        "structure_variant",
        "checkpoint_name",
        "checkpoint_update",
        "checkpoint_env_steps",
        "episodes",
        "mean_episode_return",
        "goal_rate",
        "mean_goal_count",
        "mean_pass_count",
        "mean_pass_to_shot_count",
        "mean_assist_count",
        "mean_same_owner_possession_length",
        "mean_episode_length",
        "quality_score",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name) for name in fieldnames})


def write_top_markdown(top_rows: list[dict[str, float | int | str]], path: Path) -> None:
    lines = [
        "# Phase 2 Extended Top Checkpoints",
        "",
        "| rank | condition | checkpoint | env_steps | goal_rate | pass_count | pass_to_shot | assists | possession | quality |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for idx, row in enumerate(top_rows, start=1):
        lines.append(
            "| "
            f"{idx} | "
            f"{row['condition']} | "
            f"{row['checkpoint_name']} | "
            f"{int(row['checkpoint_env_steps'])} | "
            f"{float(row['goal_rate']):.3f} | "
            f"{float(row['mean_pass_count']):.3f} | "
            f"{float(row['mean_pass_to_shot_count']):.3f} | "
            f"{float(row['mean_assist_count']):.3f} | "
            f"{float(row['mean_same_owner_possession_length']):.2f} | "
            f"{float(row['quality_score']):.2f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_best_markdown(best_rows: list[dict[str, float | int | str]], path: Path) -> None:
    lines = [
        "# Phase 2 Extended Best Checkpoints",
        "",
        "| condition | criterion | checkpoint | env_steps | goal_rate | pass_count | pass_to_shot | assists | possession |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in best_rows:
        lines.append(
            "| "
            f"{row['condition']} | "
            f"{row['criterion']} | "
            f"{row['checkpoint_name']} | "
            f"{int(row['checkpoint_env_steps'])} | "
            f"{float(row['goal_rate']):.3f} | "
            f"{float(row['mean_pass_count']):.3f} | "
            f"{float(row['mean_pass_to_shot_count']):.3f} | "
            f"{float(row['mean_assist_count']):.3f} | "
            f"{float(row['mean_same_owner_possession_length']):.2f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_ranking_plot(condition_best_rows: list[dict[str, float | int | str]], path: Path) -> None:
    ordered = sorted(condition_best_rows, key=lambda row: float(row["quality_score"]), reverse=True)
    labels = [str(row["condition"]) for row in ordered]
    scores = [float(row["quality_score"]) for row in ordered]

    fig, axis = plt.subplots(figsize=(10, 4.8))
    bars = axis.barh(labels, scores, color=["#1f77b4", "#4daf4a", "#ff7f0e", "#d62728"][: len(labels)])
    axis.invert_yaxis()
    axis.set_xlabel("Deterministic Checkpoint Quality Score")
    axis.set_title("Phase 2 Extended Condition Ranking")
    axis.grid(True, axis="x", alpha=0.25)
    for bar, row in zip(bars, ordered):
        axis.text(
            bar.get_width(),
            bar.get_y() + bar.get_height() / 2.0,
            f" p2s={float(row['mean_pass_to_shot_count']):.2f} pass={float(row['mean_pass_count']):.2f}",
            va="center",
            ha="left",
            fontsize=9,
        )
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = build_parser().parse_args()
    results_root = Path(args.results_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    requested = set(args.conditions)
    selected_conditions = [item for item in CONDITIONS if condition_label(*item) in requested]

    rows: list[dict[str, float | int | str]] = []
    for reward_variant, structure_variant in selected_conditions:
        run_dir = results_root / reward_variant / structure_variant
        checkpoint_dir = run_dir / "checkpoints"
        checkpoint_paths = sorted(checkpoint_dir.glob("update_*.pt"), key=checkpoint_sort_key)
        if args.checkpoint_stride > 1:
            checkpoint_paths = checkpoint_paths[:: int(args.checkpoint_stride)]

        for checkpoint_path in checkpoint_paths:
            summary = evaluate_variant(run_dir, checkpoint_path.name, args.episodes)
            summary["condition"] = condition_label(reward_variant, structure_variant)
            summary["reward_variant"] = reward_variant
            summary["structure_variant"] = structure_variant
            summary["checkpoint_name"] = checkpoint_path.name
            summary["quality_score"] = quality_score(summary)
            rows.append(summary)
            print(
                "[phase2_eval] "
                f"condition={summary['condition']} "
                f"checkpoint={summary['checkpoint_name']} "
                f"env_steps={summary['checkpoint_env_steps']} "
                f"goal_rate={summary['goal_rate']:.3f} "
                f"passes={summary['mean_pass_count']:.3f} "
                f"pass_to_shot={summary['mean_pass_to_shot_count']:.3f} "
                f"assists={summary['mean_assist_count']:.3f} "
                f"possession={summary['mean_same_owner_possession_length']:.2f}",
                flush=True,
            )

    rows.sort(key=lambda item: (str(item["condition"]), int(item["checkpoint_env_steps"])))
    ranking = sorted(rows, key=ranking_key, reverse=True)

    best_rows: list[dict[str, float | int | str]] = []
    condition_best_rows: list[dict[str, float | int | str]] = []
    for reward_variant, structure_variant in selected_conditions:
        condition = condition_label(reward_variant, structure_variant)
        subset = [row for row in rows if row["condition"] == condition]
        if not subset:
            continue
        best_pass_to_shot = max(subset, key=lambda row: (float(row["mean_pass_to_shot_count"]), float(row["mean_pass_count"])))
        best_pass = max(subset, key=lambda row: (float(row["mean_pass_count"]), float(row["mean_pass_to_shot_count"])))
        best_goal = max(subset, key=goal_key)
        best_overall = max(subset, key=ranking_key)
        condition_best_rows.append(best_overall)
        for criterion, row in (
            ("best_pass_to_shot", best_pass_to_shot),
            ("best_pass", best_pass),
            ("best_goal", best_goal),
            ("best_overall", best_overall),
        ):
            tagged = dict(row)
            tagged["criterion"] = criterion
            best_rows.append(tagged)

    (output_dir / "checkpoint_eval_results.json").write_text(
        json.dumps(rows, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_csv(rows, output_dir / "checkpoint_eval_results.csv")
    write_top_markdown(ranking[: args.top_k], output_dir / "top_checkpoints.md")
    write_best_markdown(best_rows, output_dir / "best_checkpoints.md")
    (output_dir / "best_checkpoints.json").write_text(
        json.dumps(best_rows, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "condition_ranking.json").write_text(
        json.dumps(condition_best_rows, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_ranking_plot(condition_best_rows, output_dir / "deterministic_condition_ranking.png")

    summary = {
        "episodes_per_checkpoint": int(args.episodes),
        "checkpoint_stride": int(args.checkpoint_stride),
        "num_conditions": len(selected_conditions),
        "num_checkpoints_evaluated": len(rows),
        "conditions": [condition_label(*item) for item in selected_conditions],
        "best_overall_checkpoint": max(rows, key=ranking_key) if rows else None,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"[phase2_eval] output_dir={output_dir}", flush=True)


if __name__ == "__main__":
    main()
