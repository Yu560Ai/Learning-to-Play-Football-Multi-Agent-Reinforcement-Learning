#!/bin/bash
cd /home/yuhan/Codes/RL/Learning-to-Play-Football-Multi-Agent-Reinforcement-Learning
export DISPLAY=""
export SDL_VIDEODRIVER=dummy
export SDL_AUDIODRIVER=dummy
.venv_yfu_grf_sys/bin/python Two_V_Two/evaluation/render_policy_video.py \
  --checkpoint Two_V_Two/results/course_runs/r2_shared_seed1/r2_progress/shared_ppo/checkpoints/latest.pt \
  --output_mp4 results/my_video.mp4 \
  --episodes 1 --fps 10 --seed 42
