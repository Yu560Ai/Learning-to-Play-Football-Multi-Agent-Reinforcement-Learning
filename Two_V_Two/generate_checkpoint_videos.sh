#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$ROOT_DIR/.venv_yfu_grf_sys/bin/python}"

"$PYTHON_BIN" "$ROOT_DIR/Two_V_Two/evaluation/generate_checkpoint_videos.py" "$@"
