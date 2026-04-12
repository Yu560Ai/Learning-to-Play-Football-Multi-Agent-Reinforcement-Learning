cat > make_side_by_side_videos.sh <<'EOF'
#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/home/yuhan/Codes/RL/Learning-to-Play-Football-Multi-Agent-Reinforcement-Learning"
cd "$REPO_ROOT"

OUT_ROOT="Two_V_Two/results/videos"

if ! command -v ffmpeg >/dev/null 2>&1; then
  echo "[ERROR] ffmpeg not found. Install ffmpeg first."
  exit 1
fi

ffmpeg -y \
  -i "$OUT_ROOT/phase1_r2_shared.mp4" \
  -i "$OUT_ROOT/phase1_r3_shared.mp4" \
  -filter_complex "[0:v]scale=960:540[left];[1:v]scale=960:540[right];[left][right]hstack=inputs=2[v]" \
  -map "[v]" \
  "$OUT_ROOT/phase1_side_by_side.mp4"

ffmpeg -y \
  -i "$OUT_ROOT/phase2_r3_shared.mp4" \
  -i "$OUT_ROOT/phase2_r3_mappo.mp4" \
  -filter_complex "[0:v]scale=960:540[left];[1:v]scale=960:540[right];[left][right]hstack=inputs=2[v]" \
  -map "[v]" \
  "$OUT_ROOT/phase2_side_by_side.mp4"

echo "[DONE] Side-by-side videos created in $OUT_ROOT"
EOF

chmod +x make_side_by_side_videos.sh