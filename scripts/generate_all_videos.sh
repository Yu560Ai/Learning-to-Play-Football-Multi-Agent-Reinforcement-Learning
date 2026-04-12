#!/bin/bash
# Generate videos for all 4 training runs

cd /home/yuhan/Codes/RL/Learning-to-Play-Football-Multi-Agent-Reinforcement-Learning

export DISPLAY=""
export SDL_VIDEODRIVER=dummy
export SDL_AUDIODRIVER=dummy

mkdir -p results/videos

echo "Generating videos for all training runs..."
echo ""

# Run 1: r2_progress/shared_ppo seed 1
echo "=== Run 1: r2_progress/shared_ppo seed 1 ==="
.venv_yfu_grf_sys/bin/python Two_V_Two/evaluation/render_policy_video.py \
  --checkpoint Two_V_Two/results/course_runs/r2_shared_seed1/r2_progress/shared_ppo/checkpoints/latest.pt \
  --output_mp4 results/videos/r2_shared_seed1_latest.mp4 \
  --episodes 1 --fps 10 --seed 42
echo "✓ Saved: results/videos/r2_shared_seed1_latest.mp4"
echo ""

# Run 2: r2_progress/shared_ppo seed 2
echo "=== Run 2: r2_progress/shared_ppo seed 2 ==="
.venv_yfu_grf_sys/bin/python Two_V_Two/evaluation/render_policy_video.py \
  --checkpoint Two_V_Two/results/course_runs/r2_shared_seed2/r2_progress/shared_ppo/checkpoints/latest.pt \
  --output_mp4 results/videos/r2_shared_seed2_latest.mp4 \
  --episodes 1 --fps 10 --seed 42
echo "✓ Saved: results/videos/r2_shared_seed2_latest.mp4"
echo ""

# Run 3: r3_assist/shared_ppo seed 1
echo "=== Run 3: r3_assist/shared_ppo seed 1 ==="
.venv_yfu_grf_sys/bin/python Two_V_Two/evaluation/render_policy_video.py \
  --checkpoint Two_V_Two/results/course_runs/r3_shared_seed1/r3_assist/shared_ppo/checkpoints/latest.pt \
  --output_mp4 results/videos/r3_shared_seed1_latest.mp4 \
  --episodes 1 --fps 10 --seed 42
echo "✓ Saved: results/videos/r3_shared_seed1_latest.mp4"
echo ""

# Run 4: r3_assist/mappo_id_cc seed 1
echo "=== Run 4: r3_assist/mappo_id_cc seed 1 ==="
.venv_yfu_grf_sys/bin/python Two_V_Two/evaluation/render_policy_video.py \
  --checkpoint Two_V_Two/results/course_runs/r3_mappo_seed1/r3_assist/mappo_id_cc/checkpoints/latest.pt \
  --output_mp4 results/videos/r3_mappo_seed1_latest.mp4 \
  --episodes 1 --fps 10 --seed 42
echo "✓ Saved: results/videos/r3_mappo_seed1_latest.mp4"
echo ""

# Also generate from middle checkpoints to show progression
echo "=== Generating progression videos (middle checkpoints) ==="

# r2_shared_seed1 progression
.venv_yfu_grf_sys/bin/python Two_V_Two/evaluation/render_policy_video.py \
  --checkpoint Two_V_Two/results/course_runs/r2_shared_seed1/r2_progress/shared_ppo/checkpoints/update_000250.pt \
  --output_mp4 results/videos/r2_shared_seed1_mid.mp4 \
  --episodes 1 --fps 10 --seed 42
echo "✓ Saved: results/videos/r2_shared_seed1_mid.mp4"

# r3_shared_seed1 progression
.venv_yfu_grf_sys/bin/python Two_V_Two/evaluation/render_policy_video.py \
  --checkpoint Two_V_Two/results/course_runs/r3_shared_seed1/r3_assist/shared_ppo/checkpoints/update_000250.pt \
  --output_mp4 results/videos/r3_shared_seed1_mid.mp4 \
  --episodes 1 --fps 10 --seed 42
echo "✓ Saved: results/videos/r3_shared_seed1_mid.mp4"

# r3_mappo_seed1 progression
.venv_yfu_grf_sys/bin/python Two_V_Two/evaluation/render_policy_video.py \
  --checkpoint Two_V_Two/results/course_runs/r3_mappo_seed1/r3_assist/mappo_id_cc/checkpoints/update_000250.pt \
  --output_mp4 results/videos/r3_mappo_seed1_mid.mp4 \
  --episodes 1 --fps 10 --seed 42
echo "✓ Saved: results/videos/r3_mappo_seed1_mid.mp4"

echo ""
echo "✅ All videos generated!"
ls -lh results/videos/
