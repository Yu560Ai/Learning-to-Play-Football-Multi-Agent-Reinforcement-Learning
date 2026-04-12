cat > phase2_r3_mappo_1m_train.sh <<'EOF'
#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/home/yuhan/Codes/RL/Learning-to-Play-Football-Multi-Agent-Reinforcement-Learning"
cd "$REPO_ROOT"

OUT_ROOT="Two_V_Two/results/phase2_r3_mappo_1m"
LOG_DIR="$OUT_ROOT/logs"
mkdir -p "$LOG_DIR"

python3 Two_V_Two/run_phase2_extended.py \
  --conditions r3_assist/mappo_id_cc \
  --n_rollout_threads 4 \
  --episode_length 400 \
  --num_env_steps 1000000 \
  --save_interval 10 \
  --output_root "$OUT_ROOT" \
  --disable_cuda \
  2>&1 | tee "$LOG_DIR/train_1m.log"

echo "[DONE] Training finished."
EOF