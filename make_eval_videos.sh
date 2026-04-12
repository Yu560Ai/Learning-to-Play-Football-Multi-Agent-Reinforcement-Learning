#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/home/yuhan/Codes/RL/Learning-to-Play-Football-Multi-Agent-Reinforcement-Learning"
cd "$REPO_ROOT"

OUT_ROOT="Two_V_Two/results/videos"
mkdir -p "$OUT_ROOT"

VIDEO_SCRIPT="Two_V_Two/evaluation/render_policy_video.py"

# =========================================================
# EDIT THESE CHECKPOINT PATHS
# =========================================================

# Phase 1 baseline
PHASE1_R2_CKPT="Two_V_Two/results/phase1_extended/budget_5000000/r2_progress/checkpoints/latest.pt"

# Phase 1 best R3 checkpoint from sweep
PHASE1_R3_CKPT="Two_V_Two/results/phase1_extended/r3_checkpoint_sweep/BEST_R3_CHECKPOINT.pt"

# Phase 2 best shared PPO checkpoint
PHASE2_R3_SHARED_CKPT="Two_V_Two/results/phase2_extended/r3_assist/shared_ppo/checkpoints/latest.pt"

# Phase 2 best MAPPO checkpoint
PHASE2_R3_MAPPO_CKPT="Two_V_Two/results/phase2_extended/r3_assist/mappo_id_cc/checkpoints/latest.pt"

# =========================================================

python3 "$VIDEO_SCRIPT" \
  --checkpoint "$PHASE1_R2_CKPT" \
  --reward_variant r2_progress \
  --structure_variant shared_ppo \
  --episodes 3 \
  --seed 1 \
  --output_path "$OUT_ROOT/phase1_r2_shared.mp4"

python3 "$VIDEO_SCRIPT" \
  --checkpoint "$PHASE1_R3_CKPT" \
  --reward_variant r3_assist \
  --structure_variant shared_ppo \
  --episodes 3 \
  --seed 1 \
  --output_path "$OUT_ROOT/phase1_r3_shared.mp4"

python3 "$VIDEO_SCRIPT" \
  --checkpoint "$PHASE2_R3_SHARED_CKPT" \
  --reward_variant r3_assist \
  --structure_variant shared_ppo \
  --episodes 3 \
  --seed 1 \
  --output_path "$OUT_ROOT/phase2_r3_shared.mp4"

python3 "$VIDEO_SCRIPT" \
  --checkpoint "$PHASE2_R3_MAPPO_CKPT" \
  --reward_variant r3_assist \
  --structure_variant mappo_id_cc \
  --episodes 3 \
  --seed 1 \
  --output_path "$OUT_ROOT/phase2_r3_mappo.mp4"

echo "[DONE] Individual videos saved under $OUT_ROOT"
