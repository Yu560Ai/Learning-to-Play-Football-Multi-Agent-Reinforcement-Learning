from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate deterministic checkpoint videos repeatedly across seed runs for one chosen condition.",
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
        "--target_env_steps",
        nargs="*",
        type=int,
        default=[],
        help="Target env steps to resolve to the nearest saved checkpoint for each seed.",
    )
    parser.add_argument(
        "--checkpoint_names",
        nargs="*",
        default=[],
        help="Explicit checkpoint filenames to render for each seed.",
    )
    parser.add_argument("--episodes", type=int, default=1, help="Deterministic rollout episodes per video.")
    parser.add_argument("--fps", type=float, default=10.0, help="Video framerate.")
    parser.add_argument("--eval_seed", type=int, default=7, help="Deterministic game-engine seed for rendering.")
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory for generated videos. Defaults under Two_V_Two/results/monitoring/<group>/videos.",
    )
    return parser


def infer_output_dir(run_root: Path) -> Path:
    return Path("Two_V_Two/results/monitoring") / run_root.name / "videos"


def load_checkpoint_payload(path: Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict):
        raise ValueError(f"Unexpected checkpoint payload type for {path}: {type(payload)}")
    return payload


def list_checkpoints(run_dir: Path) -> list[tuple[Path, int]]:
    checkpoint_dir = run_dir / "checkpoints"
    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"Missing checkpoint directory: {checkpoint_dir}")
    pairs: list[tuple[Path, int]] = []
    for checkpoint_path in sorted(checkpoint_dir.glob("update_*.pt")):
        payload = load_checkpoint_payload(checkpoint_path)
        pairs.append((checkpoint_path, int(payload["total_env_steps"])))
    if not pairs:
        raise FileNotFoundError(f"No update_*.pt checkpoints found in {checkpoint_dir}")
    return pairs


def resolve_named_checkpoint(run_dir: Path, checkpoint_name: str) -> Path:
    checkpoint_path = run_dir / "checkpoints" / checkpoint_name
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing requested checkpoint {checkpoint_name} in {run_dir / 'checkpoints'}")
    return checkpoint_path


def resolve_nearest_checkpoint(run_dir: Path, target_env_steps: int) -> tuple[Path, int]:
    checkpoint_pairs = list_checkpoints(run_dir)
    checkpoint_path, resolved_env_steps = min(
        checkpoint_pairs,
        key=lambda item: (abs(item[1] - target_env_steps), item[1]),
    )
    return checkpoint_path, int(resolved_env_steps)


def validate_seed_run(run_root: Path, seed: int) -> Path:
    run_dir = run_root / f"seed_{seed}"
    config_path = run_dir / "config.json"
    if not run_dir.exists():
        raise FileNotFoundError(f"Missing run directory for seed {seed}: {run_dir}")
    if not config_path.exists():
        raise FileNotFoundError(f"Missing config.json for seed {seed}: {config_path}")
    config = json.loads(config_path.read_text())
    scenario_name = str(config.get("scenario_name", ""))
    if scenario_name != "two_v_two_plus_goalkeepers":
        raise ValueError(
            "Monitoring workflow is restricted to the corrected custom scenario only. "
            f"Seed {seed} run has scenario_name={scenario_name!r}"
        )
    return run_dir


def render_video(
    project_root: Path,
    checkpoint_path: Path,
    output_mp4: Path,
    episodes: int,
    fps: float,
    eval_seed: int,
) -> None:
    python_bin = project_root / ".venv_yfu_grf_sys" / "bin" / "python"
    if not python_bin.exists():
        python_bin = Path(sys.executable)
    render_script = project_root / "Two_V_Two" / "evaluation" / "render_policy_video.py"
    command = [
        str(python_bin),
        str(render_script),
        "--checkpoint",
        str(checkpoint_path),
        "--output_mp4",
        str(output_mp4),
        "--episodes",
        str(episodes),
        "--fps",
        str(fps),
        "--seed",
        str(eval_seed),
    ]
    subprocess.run(command, check=True)


def main() -> None:
    args = build_parser().parse_args()
    if not args.target_env_steps and not args.checkpoint_names:
        raise ValueError("Specify at least one --target_env_steps value or one --checkpoint_names value.")

    project_root = _project_root()
    run_root = Path(args.run_root)
    output_dir = Path(args.output_dir) if args.output_dir else infer_output_dir(run_root)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict[str, Any]] = []

    for seed in args.seeds:
        run_dir = validate_seed_run(run_root, int(seed))

        for target_env_steps in args.target_env_steps:
            checkpoint_path, resolved_env_steps = resolve_nearest_checkpoint(run_dir, int(target_env_steps))
            target_dir = output_dir / f"target_env_{int(target_env_steps):07d}"
            target_dir.mkdir(parents=True, exist_ok=True)
            output_mp4 = target_dir / f"seed_{seed}_2d.mp4"
            render_video(project_root, checkpoint_path, output_mp4, args.episodes, args.fps, args.eval_seed)
            summary_rows.append(
                {
                    "seed": int(seed),
                    "request_type": "target_env_steps",
                    "requested_value": int(target_env_steps),
                    "resolved_checkpoint": checkpoint_path.name,
                    "resolved_env_steps": int(resolved_env_steps),
                    "run_dir": str(run_dir),
                    "video_path": str(output_mp4),
                    "metadata_path": str(output_mp4.with_suffix(".json")),
                }
            )
            print(
                "[monitor_videos] "
                f"seed={seed} "
                f"target_env_steps={int(target_env_steps)} "
                f"resolved={checkpoint_path.name} "
                f"resolved_env_steps={int(resolved_env_steps)} "
                f"video={output_mp4}",
                flush=True,
            )

        for checkpoint_name in args.checkpoint_names:
            checkpoint_path = resolve_named_checkpoint(run_dir, checkpoint_name)
            payload = load_checkpoint_payload(checkpoint_path)
            resolved_env_steps = int(payload["total_env_steps"])
            target_dir = output_dir / checkpoint_name.replace(".pt", "")
            target_dir.mkdir(parents=True, exist_ok=True)
            output_mp4 = target_dir / f"seed_{seed}_2d.mp4"
            render_video(project_root, checkpoint_path, output_mp4, args.episodes, args.fps, args.eval_seed)
            summary_rows.append(
                {
                    "seed": int(seed),
                    "request_type": "checkpoint_name",
                    "requested_value": checkpoint_name,
                    "resolved_checkpoint": checkpoint_path.name,
                    "resolved_env_steps": int(resolved_env_steps),
                    "run_dir": str(run_dir),
                    "video_path": str(output_mp4),
                    "metadata_path": str(output_mp4.with_suffix(".json")),
                }
            )
            print(
                "[monitor_videos] "
                f"seed={seed} "
                f"checkpoint={checkpoint_name} "
                f"resolved_env_steps={int(resolved_env_steps)} "
                f"video={output_mp4}",
                flush=True,
            )

    summary = {
        "run_root": str(run_root),
        "seeds": [int(seed) for seed in args.seeds],
        "episodes": int(args.episodes),
        "fps": float(args.fps),
        "eval_seed": int(args.eval_seed),
        "videos": summary_rows,
    }
    (output_dir / "video_index.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"[monitor_videos] index={output_dir / 'video_index.json'}", flush=True)


if __name__ == "__main__":
    main()
