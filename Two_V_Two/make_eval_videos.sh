#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$ROOT_DIR/.venv_yfu_grf_sys/bin/python}"
VIDEO_DIR="$ROOT_DIR/Two_V_Two/results/videos"

mkdir -p "$VIDEO_DIR"

# Phase 1
# R2: stable long-budget baseline
PHASE1_R2_CKPT="$ROOT_DIR/Two_V_Two/results/phase1_extended/budget_5000000/r2_progress/checkpoints/latest.pt"
# R3: best deterministic pass checkpoint from results/phase1_extended/r3_checkpoint_sweep/summary.json
PHASE1_R3_CKPT="$ROOT_DIR/Two_V_Two/results/phase1_extended/budget_5000000/r3_assist/checkpoints/update_000150.pt"

# Phase 2
# Deterministic phase-2 sweep never found nonzero pass/pass_to_shot checkpoints.
# These are the best_overall deterministic checkpoints from results/phase2_extended/analysis/best_checkpoints.json.
PHASE2_R3_SHARED_CKPT="$ROOT_DIR/Two_V_Two/results/phase2_extended/r3_assist/shared_ppo/checkpoints/update_000120.pt"
PHASE2_R3_MAPPO_CKPT="$ROOT_DIR/Two_V_Two/results/phase2_extended/r3_assist/mappo_id_cc/checkpoints/update_000280.pt"

COMMON_ARGS=(
  --episodes 1
  --fps 10
  --seed 7
)

"$PYTHON_BIN" "$ROOT_DIR/Two_V_Two/evaluation/render_policy_video.py" \
  --checkpoint "$PHASE1_R2_CKPT" \
  --output_mp4 "$VIDEO_DIR/phase1_r2_progress_baseline.mp4" \
  "${COMMON_ARGS[@]}"

"$PYTHON_BIN" "$ROOT_DIR/Two_V_Two/evaluation/render_policy_video.py" \
  --checkpoint "$PHASE1_R3_CKPT" \
  --output_mp4 "$VIDEO_DIR/phase1_r3_assist_best.mp4" \
  "${COMMON_ARGS[@]}"

"$PYTHON_BIN" "$ROOT_DIR/Two_V_Two/evaluation/render_policy_video.py" \
  --checkpoint "$PHASE2_R3_SHARED_CKPT" \
  --output_mp4 "$VIDEO_DIR/phase2_r3_assist_shared_ppo_best.mp4" \
  "${COMMON_ARGS[@]}"

"$PYTHON_BIN" "$ROOT_DIR/Two_V_Two/evaluation/render_policy_video.py" \
  --checkpoint "$PHASE2_R3_MAPPO_CKPT" \
  --output_mp4 "$VIDEO_DIR/phase2_r3_assist_mappo_id_cc_best.mp4" \
  "${COMMON_ARGS[@]}"

echo "[videos] individual outputs saved under $VIDEO_DIR"
