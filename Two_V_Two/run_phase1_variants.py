from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


VARIANTS = [
    "r1_scoring",
    "r2_progress",
    "r3_assist",
    "r4_anti_selfish",
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run Phase 1 reward-variant pilots with the shared-policy PPO baseline.",
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=VARIANTS,
        default=VARIANTS,
        help="Reward variants to run.",
    )
    parser.add_argument("--num_env_steps", type=int, default=256)
    parser.add_argument("--episode_length", type=int, default=64)
    parser.add_argument("--n_rollout_threads", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--save_interval", type=int, default=1)
    parser.add_argument(
        "--output_root",
        type=str,
        default="Two_V_Two/results/phase1",
        help="Directory root for per-variant outputs.",
    )
    parser.add_argument("--progress_reward_coef", type=float, default=0.05)
    parser.add_argument("--assist_reward", type=float, default=0.5)
    parser.add_argument("--assist_window", type=int, default=25)
    parser.add_argument("--selfish_possession_threshold", type=int, default=12)
    parser.add_argument("--selfish_penalty", type=float, default=0.02)
    parser.add_argument(
        "--disable_cuda",
        action="store_true",
        default=False,
        help="Forward --disable_cuda to the trainer.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    for variant in args.variants:
        run_dir = output_root / variant
        run_dir.mkdir(parents=True, exist_ok=True)
        command = [
            sys.executable,
            "Two_V_Two/train_basic.py",
            "--reward_variant",
            variant,
            "--num_env_steps",
            str(args.num_env_steps),
            "--episode_length",
            str(args.episode_length),
            "--n_rollout_threads",
            str(args.n_rollout_threads),
            "--seed",
            str(args.seed),
            "--save_interval",
            str(args.save_interval),
            "--run_dir",
            str(run_dir),
            "--progress_reward_coef",
            str(args.progress_reward_coef),
            "--assist_reward",
            str(args.assist_reward),
            "--assist_window",
            str(args.assist_window),
            "--selfish_possession_threshold",
            str(args.selfish_possession_threshold),
            "--selfish_penalty",
            str(args.selfish_penalty),
        ]
        if args.disable_cuda:
            command.append("--disable_cuda")

        print(f"[phase1] running {variant} -> {run_dir}", flush=True)
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
