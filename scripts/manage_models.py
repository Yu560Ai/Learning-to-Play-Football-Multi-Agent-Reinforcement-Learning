#!/usr/bin/env python3
"""Manage best model artifacts and metadata in a team workflow."""

import argparse
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

BEST_MODELS_DIR = Path(__file__).resolve().parents[1] / "best_models"
TEAM_README = Path(__file__).resolve().parents[1] / "README_TEAM.md"
MODEL_TABLE_HEADER = "| File | Algo | Scenario | Owner | Winrate | Notes | Size |\n|---|---|---|---|---|---|---|\n"
MAX_SIZE_BYTES = 95 * 1024 * 1024


def ensure_best_models_dir():
    BEST_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    marker = BEST_MODELS_DIR / ".gitkeep"
    marker.touch(exist_ok=True)


def sanitize_tag(value: str) -> str:
    value = value.strip().replace(" ", "_")
    value = re.sub(r"[^a-zA-Z0-9_\-]", "", value)
    return value


def format_model_filename(algo, scenario, owner, winrate):
    algo_tag = sanitize_tag(algo)
    scenario_tag = sanitize_tag(scenario)
    owner_tag = sanitize_tag(owner)
    winrate_tag = str(winrate).replace("%", "")
    if not winrate_tag.replace('.', '').isdigit():
        raise ValueError("winrate must be numeric or numeric with percent symbol")
    winrate_tag = winrate_tag.strip()
    return f"{algo_tag}_{scenario_tag}_{owner_tag}_{winrate_tag}%.pt"


def load_current_table():
    if not TEAM_README.exists():
        return None
    content = TEAM_README.read_text(encoding="utf-8")
    if "| File |" not in content:
        return content + "\n## Model Registry\n\n" + MODEL_TABLE_HEADER
    return content


def update_readme_table(filename, algo, scenario, owner, winrate, notes, size_bytes):
    size_mb = size_bytes / (1024.0 * 1024.0)
    size_str = f"{size_mb:.2f} MB"
    content = load_current_table()
    if content is None:
        content = "# Team model registry\n\n## Model Registry\n\n" + MODEL_TABLE_HEADER
    if "| File | Algo |" not in content:
        content += "\n## Model Registry\n\n" + MODEL_TABLE_HEADER

    before, table = content.split("| File |", 1)
    table = "| File |" + table
    lines = table.strip().splitlines()
    if len(lines) <= 2:
        lines = ["| File | Algo | Scenario | Owner | Winrate | Notes | Size |", "|---|---|---|---|---|---|---|"]
    new_row = f"| [{filename}](best_models/{filename}) | {algo} | {scenario} | {owner} | {winrate}% | {notes} | {size_str} |"
    lines.append(new_row)

    new_table = "\n".join(lines)
    TEAM_README.write_text(before + new_table + "\n", encoding="utf-8")


def compress_model(source: Path, target: Path):
    # Quick heuristic: copy and rely on model author to structure checkpoint accordingly.
    # Since we cannot inspect arbitrary `.pt` contents robustly, we drop optimizer state if present in popular keys.
    import torch

    checkpoint = torch.load(source, map_location="cpu")
    if isinstance(checkpoint, dict):
        keys = list(checkpoint.keys())
        optimizer_keys = [k for k in keys if "optim" in k.lower() or "optimizer" in k.lower()]
        if optimizer_keys:
            for k in optimizer_keys:
                checkpoint.pop(k, None)
        if "optimizer_state_dict" in checkpoint:
            checkpoint.pop("optimizer_state_dict", None)
        if "epoch" in checkpoint:
            # keep epoch for reproducibility
            pass
    torch.save(checkpoint, target)


def stage_files(paths):
    try:
        subprocess.run(["git", "add", *paths], check=True)
        print("Staged for git:", ", ".join(str(p) for p in paths))
    except subprocess.CalledProcessError as exc:
        print("Error staging files:", exc)


def add_model(args):
    ensure_best_models_dir()
    filename = format_model_filename(args.algo, args.scenario, args.name, args.winrate)
    dest_path = BEST_MODELS_DIR / filename

    if not args.source:
        raise ValueError("--source is required to copy model weights into best_models.")
    source_path = Path(args.source)
    if not source_path.exists():
        raise FileNotFoundError(f"Source file does not exist: {source_path}")

    if source_path.resolve() == dest_path.resolve():
        raise ValueError("Source and destination are the same file")

    shutil.copy2(source_path, dest_path)
    print(f"Copied {source_path} -> {dest_path}")

    size_bytes = dest_path.stat().st_size
    if size_bytes > MAX_SIZE_BYTES:
        compressed_path = BEST_MODELS_DIR / f"{dest_path.stem}.compressed.pt"
        compress_model(dest_path, compressed_path)
        compressed_size = compressed_path.stat().st_size
        if compressed_size < size_bytes:
            dest_path.unlink()
            compressed_path.rename(dest_path)
            size_bytes = compressed_size
            print(f"Compressed model from {size_bytes:.2f} bytes to {compressed_size:.2f} bytes")
        else:
            compressed_path.unlink()
            print("Compression did not reduce size; using original file")

    update_readme_table(filename, args.algo, args.scenario, args.name, args.winrate, args.notes, size_bytes)

    if args.commit:
        stage_files([dest_path, TEAM_README])

    print(f"Model added: {filename}")


def parse_args():
    parser = argparse.ArgumentParser(description="Manage best_models entries and metadata")
    parser.add_argument("--add", action="store_true", help="Add a model snapshot to best_models")
    parser.add_argument("--source", type=str, help="Path of the source checkpoint file")
    parser.add_argument("--name", type=str, required="--add" in sys.argv, help="Team member name")
    parser.add_argument("--algo", type=str, required="--add" in sys.argv, help="Algorithm name")
    parser.add_argument("--scenario", type=str, required="--add" in sys.argv, help="Scenario name")
    parser.add_argument("--winrate", type=float, required="--add" in sys.argv, help="Winrate as percentage number")
    parser.add_argument("--notes", type=str, default="", help="Optional notes")
    parser.add_argument("--commit", action="store_true", help="Stage added files for git commit")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.add:
        try:
            add_model(args)
        except Exception as exc:
            print("Error adding model:", exc)
            sys.exit(1)
    else:
        print("No action specified. Use --add with required options.")
        sys.exit(0)


if __name__ == "__main__":
    main()
