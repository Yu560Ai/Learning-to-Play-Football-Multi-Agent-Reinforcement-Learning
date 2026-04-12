#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$ROOT_DIR/.venv_yfu_grf_sys/bin/python}"
VIDEO_DIR="$ROOT_DIR/Two_V_Two/results/videos"

mkdir -p "$VIDEO_DIR"

if command -v ffmpeg >/dev/null 2>&1; then
  FFMPEG_BIN="$(command -v ffmpeg)"
else
  FFMPEG_BIN="$("$PYTHON_BIN" -c 'import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())')"
fi

if [[ ! -x "$FFMPEG_BIN" ]]; then
  echo "ffmpeg binary not found. Install ffmpeg or imageio-ffmpeg first." >&2
  exit 1
fi

PHASE1_LEFT="$VIDEO_DIR/phase1_r2_progress_baseline.mp4"
PHASE1_RIGHT="$VIDEO_DIR/phase1_r3_assist_best.mp4"
PHASE1_OUT="$VIDEO_DIR/phase1_r2_vs_r3_side_by_side.mp4"

PHASE2_LEFT="$VIDEO_DIR/phase2_r3_assist_shared_ppo_best.mp4"
PHASE2_RIGHT="$VIDEO_DIR/phase2_r3_assist_mappo_id_cc_best.mp4"
PHASE2_OUT="$VIDEO_DIR/phase2_r3_shared_vs_mappo_side_by_side.mp4"

for input_path in "$PHASE1_LEFT" "$PHASE1_RIGHT" "$PHASE2_LEFT" "$PHASE2_RIGHT"; do
  if [[ ! -f "$input_path" ]]; then
    echo "missing input video: $input_path" >&2
    exit 1
  fi
done

"$FFMPEG_BIN" -y \
  -i "$PHASE1_LEFT" \
  -i "$PHASE1_RIGHT" \
  -filter_complex "[0:v][1:v]hstack=inputs=2[v]" \
  -map "[v]" \
  -an \
  -c:v libx264 \
  -pix_fmt yuv420p \
  -movflags +faststart \
  "$PHASE1_OUT"

"$FFMPEG_BIN" -y \
  -i "$PHASE2_LEFT" \
  -i "$PHASE2_RIGHT" \
  -filter_complex "[0:v][1:v]hstack=inputs=2[v]" \
  -map "[v]" \
  -an \
  -c:v libx264 \
  -pix_fmt yuv420p \
  -movflags +faststart \
  "$PHASE2_OUT"

echo "[videos] comparison outputs saved under $VIDEO_DIR"
