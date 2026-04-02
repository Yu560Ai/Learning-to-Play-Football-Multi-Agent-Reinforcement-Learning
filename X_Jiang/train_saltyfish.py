from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _ensure_repo_on_path() -> None:
    root = _repo_root()
    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)


PRESETS: dict[str, list[str]] = {
    "five_v_five_quick": [
        "--env-name",
        "5_vs_5",
        "--num-controlled-players",
        "4",
        "--total-timesteps",
        "40960",
        "--rollout-steps",
        "512",
        "--num-minibatches",
        "8",
        "--learning-rate",
        "1e-4",
        "--gamma",
        "0.993",
        "--save-interval",
        "10",
        "--save-dir",
        "X_Jiang/checkpoints/saltyfish_five_v_five",
        "--logdir",
        "X_Jiang/logs/saltyfish_five_v_five",
    ],
    "five_v_five_base": [
        "--env-name",
        "5_vs_5",
        "--num-controlled-players",
        "4",
        "--total-timesteps",
        "204800",
        "--rollout-steps",
        "1024",
        "--num-minibatches",
        "8",
        "--learning-rate",
        "1e-4",
        "--gamma",
        "0.993",
        "--save-interval",
        "20",
        "--save-dir",
        "X_Jiang/checkpoints/saltyfish_five_v_five",
        "--logdir",
        "X_Jiang/logs/saltyfish_five_v_five",
    ],
}


def parse_args() -> tuple[str, list[str]]:
    parser = argparse.ArgumentParser(description="Train X_Jiang SaltyFish-style 5v5 baseline.")
    parser.add_argument(
        "--preset",
        default="five_v_five_quick",
        choices=sorted(PRESETS.keys()),
        help="Training preset.",
    )
    args, extras = parser.parse_known_args()
    return args.preset, extras


def main() -> None:
    preset_name, extras = parse_args()
    _ensure_repo_on_path()
    from X_Jiang.saltyfish_five_baseline.ppo import main as salty_main

    salty_main(PRESETS[preset_name] + extras)


if __name__ == "__main__":
    main()
