from __future__ import annotations

import sys
from pathlib import Path
import argparse


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _ensure_repo_on_path() -> None:
    root = _repo_root()
    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate X_Jiang SaltyFish-style 5v5 baseline.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--compare-random", action="store_true")
    parser.add_argument("--save-video", action="store_true")
    parser.add_argument("--video-dir", default="X_Jiang/videos/saltyfish_five_v_five")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _ensure_repo_on_path()
    from X_Jiang.saltyfish_five_baseline.evaluate import main as salty_eval_main

    argv = [
        "--checkpoint",
        args.checkpoint,
        "--episodes",
        str(args.episodes),
        "--device",
        args.device,
        "--video-dir",
        args.video_dir,
    ]
    if args.deterministic:
        argv.append("--deterministic")
    if args.render:
        argv.append("--render")
    if args.compare_random:
        argv.append("--compare-random")
    if args.save_video:
        argv.append("--save-video")
    salty_eval_main(argv)


if __name__ == "__main__":
    main()
