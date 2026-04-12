# FINAL_TRAIN.md

## Purpose

This document records the **exact training and evaluation pipeline** used to produce the final results shown in `FINAL.md`.

All paths are **absolute relative to repository root**:

```
~/Codes/RL/Learning-to-Play-Football-Multi-Agent-Reinforcement-Learning
```

---

# 1. Custom Environment Setup (2v2 + Goalkeepers)

## Scenario Definition

We use a custom GRF scenario:

```
football-master/gfootball/scenarios/two_v_two_plus_goalkeepers.py
```

### Configuration

* Left team:

  * 2 controlled players
  * 1 built-in goalkeeper
* Right team:

  * 2 built-in players
  * 1 built-in goalkeeper

---

## Environment Usage in Training

All training runs use:

```
env_name = "two_v_two_plus_goalkeepers"
```

inside:

```
Two_V_Two/grf_simple_env.py
```

---

# 2. Training Runs

All training is launched using:

```
Two_V_Two/run_phase2_extended.py
```

---

## Parallel Training Commands

Executed from repository root:

```bash
python3 Two_V_Two/run_phase2_extended.py \
  --conditions r2_progress/shared_ppo \
  --seed 1 \
  --n_rollout_threads 4 \
  --episode_length 400 \
  --num_env_steps 2000000 \
  --save_interval 10 \
  --output_root Two_V_Two/results/course_runs/r2_shared_seed1 \
  --disable_cuda &

python3 Two_V_Two/run_phase2_extended.py \
  --conditions r2_progress/shared_ppo \
  --seed 2 \
  --n_rollout_threads 4 \
  --episode_length 400 \
  --num_env_steps 2000000 \
  --save_interval 10 \
  --output_root Two_V_Two/results/course_runs/r2_shared_seed2 \
  --disable_cuda &

python3 Two_V_Two/run_phase2_extended.py \
  --conditions r3_assist/shared_ppo \
  --seed 1 \
  --n_rollout_threads 4 \
  --episode_length 400 \
  --num_env_steps 2000000 \
  --save_interval 10 \
  --output_root Two_V_Two/results/course_runs/r3_shared_seed1 \
  --disable_cuda &

python3 Two_V_Two/run_phase2_extended.py \
  --conditions r3_assist/mappo_id_cc \
  --seed 1 \
  --n_rollout_threads 4 \
  --episode_length 400 \
  --num_env_steps 2000000 \
  --save_interval 10 \
  --output_root Two_V_Two/results/course_runs/r3_mappo_seed1 \
  --disable_cuda &
```

---

## Training Output Structure

Each run produces:

```
Two_V_Two/results/course_runs/<run_name>/
    └── <reward>/<structure>/
        ├── checkpoints/
        ├── metrics.jsonl
        └── config.json
```

Examples:

```
Two_V_Two/results/course_runs/r2_shared_seed1/r2_progress/shared_ppo/
Two_V_Two/results/course_runs/r3_shared_seed1/r3_assist/shared_ppo/
Two_V_Two/results/course_runs/r3_mappo_seed1/r3_assist/mappo_id_cc/
```

---

# 3. Monitoring Group Setup (for Video + Curves)

To unify different runs under a common interface, we created **monitor groups**:

```
Two_V_Two/results/monitor_groups/
```

---

## Symlink Commands

```bash
mkdir -p Two_V_Two/results/monitor_groups/r2_shared
mkdir -p Two_V_Two/results/monitor_groups/r3_shared
mkdir -p Two_V_Two/results/monitor_groups/r3_mappo
```

### R2 shared (2 seeds)

```bash
ln -sfn ../../course_runs/r2_shared_seed1/r2_progress/shared_ppo \
Two_V_Two/results/monitor_groups/r2_shared/seed_1

ln -sfn ../../course_runs/r2_shared_seed2/r2_progress/shared_ppo \
Two_V_Two/results/monitor_groups/r2_shared/seed_2
```

### R3 shared

```bash
ln -sfn ../../course_runs/r3_shared_seed1/r3_assist/shared_ppo \
Two_V_Two/results/monitor_groups/r3_shared/seed_1
```

### R3 MAPPO

```bash
ln -sfn ../../course_runs/r3_mappo_seed1/r3_assist/mappo_id_cc \
Two_V_Two/results/monitor_groups/r3_mappo/seed_1
```

---

# 4. Learning Curve Generation

Script:

```
Two_V_Two/refresh_learning_curves.sh
```

---

## Commands

### R2 (multi-seed)

```bash
Two_V_Two/refresh_learning_curves.sh \
  --run_root Two_V_Two/results/monitor_groups/r2_shared \
  --seeds 1 2
```

### R3 shared

```bash
Two_V_Two/refresh_learning_curves.sh \
  --run_root Two_V_Two/results/monitor_groups/r3_shared \
  --seeds 1
```

### R3 MAPPO

```bash
Two_V_Two/refresh_learning_curves.sh \
  --run_root Two_V_Two/results/monitor_groups/r3_mappo \
  --seeds 1
```

---

## Outputs

Each produces:

```
learning_curves.png
curve_summary.json
```

---

# 5. Video Generation

Script:

```
Two_V_Two/generate_checkpoint_videos.sh
```

Renderer:

```
Two_V_Two/evaluation/render_policy_video.py
```

The video renderer uses the **default GRF video output**:

* video is dumped directly by the GRF environment during deterministic evaluation
* the visual style matches the standard GRF board-style episode video
* for this `2v2 + goalkeepers` setup, the action table shows only the 2 controlled players
* the dumped default video is converted to the final `*_2d.mp4` file for reporting

---

## Selected Checkpoints

We use:

* `update_000100.pt`
* `update_000250.pt`
* `update_000490.pt`

---

## Commands

### R2 shared

```bash
Two_V_Two/generate_checkpoint_videos.sh \
  --run_root Two_V_Two/results/monitor_groups/r2_shared \
  --seeds 1 2 \
  --checkpoint_names update_000100.pt update_000250.pt update_000490.pt \
  --episodes 1
```

---

### R3 shared

```bash
Two_V_Two/generate_checkpoint_videos.sh \
  --run_root Two_V_Two/results/monitor_groups/r3_shared \
  --seeds 1 \
  --checkpoint_names update_000100.pt update_000250.pt update_000490.pt \
  --episodes 1
```

---

### R3 MAPPO

```bash
Two_V_Two/generate_checkpoint_videos.sh \
  --run_root Two_V_Two/results/monitor_groups/r3_mappo \
  --seeds 1 \
  --checkpoint_names update_000100.pt update_000250.pt update_000490.pt \
  --episodes 1
```

---

## Output Locations

Videos are saved under:

```
Two_V_Two/results/monitoring/<group>/videos/<checkpoint>/seed_X_2d.mp4
```

Examples:

```
Two_V_Two/results/monitoring/r2_shared/videos/update_000250/seed_1_2d.mp4
Two_V_Two/results/monitoring/r2_shared/videos/update_000250/seed_2_2d.mp4
Two_V_Two/results/monitoring/r3_shared/videos/update_000250/seed_1_2d.mp4
Two_V_Two/results/monitoring/r3_mappo/videos/update_000250/seed_1_2d.mp4
```

For reporting convenience, the final 12 selected MP4 files are also copied into:

```
Two_V_Two/results/Final_video/
```

---

# 6. Process Management (Stopping Training)

To terminate all training processes:

```bash
pkill -9 -f train_basic.py
```

---

# Final Notes

* All results are generated under the **custom scenario**, not GRF academy tasks.
* Training uses:

  * shared PPO
  * MAPPO centralized critic
* Evaluation is based on:

  * learning curves
  * rollout videos at selected checkpoints

---
