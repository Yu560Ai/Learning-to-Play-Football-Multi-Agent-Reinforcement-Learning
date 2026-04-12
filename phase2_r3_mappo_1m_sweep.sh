#!/usr/bin/env bash
set -euo pipefail

# Run from repo root:
#   chmod +x phase2_r3_mappo_1m_sweep.sh
#   ./phase2_r3_mappo_1m_sweep.sh

REPO_ROOT="/home/yuhan/Codes/RL/Learning-to-Play-Football-Multi-Agent-Reinforcement-Learning"
cd "$REPO_ROOT"

RESULTS_ROOT="Two_V_Two/results/phase2_r3_mappo_1m"
ANALYSIS_DIR="$RESULTS_ROOT/analysis"
LOG_DIR="$RESULTS_ROOT/logs"
mkdir -p "$ANALYSIS_DIR" "$LOG_DIR"

EVAL_SCRIPT="Two_V_Two/evaluation/run_phase2_checkpoint_sweep.py"

if [[ ! -f "$EVAL_SCRIPT" ]]; then
  echo "[ERROR] Eval script not found: $EVAL_SCRIPT"
  exit 1
fi

python3 "$EVAL_SCRIPT" \
  --results_root "$RESULTS_ROOT" \
  --episodes 20 \
  --checkpoint_stride 1 \
  --output_dir "$ANALYSIS_DIR" \
  2>&1 | tee "$LOG_DIR/checkpoint_sweep.log"

echo "[DONE] Checkpoint sweep finished."
echo "[INFO] Analysis artifacts should be in: $ANALYSIS_DIR"
