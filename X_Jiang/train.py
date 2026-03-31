from __future__ import annotations

import argparse

from presets import PRESETS
from xjiang_football.ppo import main as ppo_main


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train X_Jiang football RL agent.")
    parser.add_argument(
        "--preset",
        type=str,
        default="five_v_five_debug",
        choices=sorted(PRESETS.keys()),
        help="Preset configuration name.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Override device, e.g. cpu or cuda. Default uses preset/auto.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    ppo_main(preset_name=args.preset, device_override=args.device)