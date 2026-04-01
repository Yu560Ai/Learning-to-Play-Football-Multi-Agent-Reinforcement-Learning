# Offline RL Commands

This file contains the updated command flow for the hybrid line:

1. Academy PPO for primitive learning
2. `5_vs_5` PPO for transfer
3. `5_vs_5` offline RL for replay-based improvement
4. PPO fine-tuning from the best IQL checkpoint

Important rule:

- do not mix Academy and `5_vs_5` datasets in the same IQL training run

## Environment

From the repository root:

```bash
cd ~/Codes/RL/Learning-to-Play-Football-Multi-Agent-Reinforcement-Learning
source football-master/football-env/bin/activate
```

## 1. Academy PPO Warm-Up

Use Academy as a capped warm-up, not an open-ended compute sink.

### Stage A: `academy_pass_and_shoot_with_keeper`

```bash
python3 Y_Fu/train.py \
  --preset academy_pass_and_shoot_with_keeper \
  --total-timesteps 800000 \
  --device cpu
```

Evaluate the chosen checkpoint:

```bash
python3 Y_Fu/evaluate.py \
  --checkpoint <best_stage2_checkpoint> \
  --episodes 20 \
  --deterministic \
  --device cpu \
  --seed 123
```

### Stage B: `academy_3_vs_1_with_keeper`

Start from the best `academy_pass_and_shoot_with_keeper` checkpoint:

```bash
python3 Y_Fu/train.py \
  --preset academy_3_vs_1_with_keeper \
  --total-timesteps 1500000 \
  --device cpu \
  --init-checkpoint <best_stage2_checkpoint>
```

Evaluate the chosen checkpoint:

```bash
python3 Y_Fu/evaluate.py \
  --checkpoint <best_stage3_checkpoint> \
  --episodes 20 \
  --deterministic \
  --device cpu \
  --seed 123
```

## 2. `5_vs_5` PPO Transfer

Start `5_vs_5` from the best Academy checkpoint:

```bash
python3 Y_Fu/train.py \
  --preset five_vs_five \
  --device cpu \
  --total-timesteps 1000000 \
  --init-checkpoint <best_academy_checkpoint>
```

Continue the early transfer check to roughly `500k env steps`:

```bash
python3 Y_Fu/train.py \
  --preset five_vs_five \
  --device cpu \
  --total-timesteps 2000000 \
  --init-checkpoint <best_stage4_checkpoint>
```

Evaluate two candidate PPO checkpoints after the early transfer run:

Best candidate:

```bash
python3 Y_Fu/evaluate.py \
  --checkpoint <best_stage4_checkpoint> \
  --episodes 20 \
  --deterministic \
  --device cpu \
  --seed 123
```

Weaker candidate:

```bash
python3 Y_Fu/evaluate.py \
  --checkpoint Y_Fu/checkpoints/five_vs_five/update_10.pt \
  --episodes 20 \
  --deterministic \
  --device cpu \
  --seed 123
```

## 3. Pilot `5_vs_5` Offline Collection From PPO

Recommended pilot total:

- `300K ~ 500K` env-steps

These pilot datasets are all `5_vs_5`. That is intentional.

### Pilot A: Best PPO, `epsilon = 0.0`

```bash
python3 Y_Fu/collect_offline_data.py \
  --checkpoint Y_Fu/checkpoints/five_vs_five/latest.pt \
  --policy checkpoint \
  --num-envs 6 \
  --total-env-steps 200000 \
  --epsilon 0.0 \
  --chunk-size 500000 \
  --save-dir Y_Fu/offline_data/pilot_5v5_best_eps0 \
  --seed 123 \
  --obs-dtype float16 \
  --checkpoint-id 1
```

### Pilot B: Best PPO, `epsilon = 0.15`

```bash
python3 Y_Fu/collect_offline_data.py \
  --checkpoint Y_Fu/checkpoints/five_vs_five/latest.pt \
  --policy checkpoint \
  --num-envs 6 \
  --total-env-steps 100000 \
  --epsilon 0.15 \
  --chunk-size 500000 \
  --save-dir Y_Fu/offline_data/pilot_5v5_best_eps15 \
  --seed 124 \
  --obs-dtype float16 \
  --checkpoint-id 1
```

### Pilot C: Weaker PPO, `epsilon = 0.05`

```bash
python3 Y_Fu/collect_offline_data.py \
  --checkpoint Y_Fu/checkpoints/five_vs_five/update_10.pt \
  --policy checkpoint \
  --num-envs 6 \
  --total-env-steps 100000 \
  --epsilon 0.05 \
  --chunk-size 500000 \
  --save-dir Y_Fu/offline_data/pilot_5v5_weaker_eps005 \
  --seed 125 \
  --obs-dtype float16 \
  --checkpoint-id 2
```

### Pilot D: Random Policy

```bash
python3 Y_Fu/collect_offline_data.py \
  --policy random \
  --num-envs 6 \
  --total-env-steps 100000 \
  --epsilon 0.0 \
  --chunk-size 500000 \
  --save-dir Y_Fu/offline_data/pilot_5v5_random \
  --seed 126 \
  --obs-dtype float16 \
  --checkpoint-id 3
```

## 4. Pilot IQL On `5_vs_5`

Start with a short smoke run:

- `20000` gradient steps
- evaluate every `5000`
- reward normalization enabled
- each loaded chunk is reused for `32` sampled batches before switching

```bash
python3 Y_Fu/train_iql.py \
  --dataset-dirs \
    Y_Fu/offline_data/pilot_5v5_best_eps0 \
    Y_Fu/offline_data/pilot_5v5_best_eps15 \
    Y_Fu/offline_data/pilot_5v5_weaker_eps005 \
    Y_Fu/offline_data/pilot_5v5_random \
  --reward-key reward \
  --device cuda \
  --save-dir Y_Fu/checkpoints/iql_5v5_pilot \
  --batch-size 4096 \
  --chunk-reuse-batches 32 \
  --learning-rate 3e-4 \
  --gamma 0.993 \
  --expectile 0.7 \
  --temperature 3.0 \
  --tau 0.005 \
  --normalize-rewards \
  --total-gradient-steps 20000 \
  --eval-interval 5000 \
  --save-interval 10000 \
  --eval-episodes 20 \
  --seed 123
```

Evaluate the pilot IQL checkpoint:

```bash
python3 Y_Fu/evaluate_iql.py \
  --checkpoint Y_Fu/checkpoints/iql_5v5_pilot/best.pt \
  --episodes 20 \
  --deterministic \
  --device cuda \
  --seed 123
```

## 5. Full `5_vs_5` IQL Training

Run this only after the pilot succeeds.

### Full collection from PPO

Best PPO, `epsilon = 0.0`:

```bash
python3 Y_Fu/collect_offline_data.py \
  --checkpoint Y_Fu/checkpoints/five_vs_five/latest.pt \
  --policy checkpoint \
  --num-envs 6 \
  --total-env-steps 20000000 \
  --epsilon 0.0 \
  --chunk-size 500000 \
  --save-dir Y_Fu/offline_data/run_5v5_best_eps0 \
  --seed 223 \
  --obs-dtype float16 \
  --checkpoint-id 1
```

Best PPO, `epsilon = 0.15`:

```bash
python3 Y_Fu/collect_offline_data.py \
  --checkpoint Y_Fu/checkpoints/five_vs_five/latest.pt \
  --policy checkpoint \
  --num-envs 6 \
  --total-env-steps 12500000 \
  --epsilon 0.15 \
  --chunk-size 500000 \
  --save-dir Y_Fu/offline_data/run_5v5_best_eps15 \
  --seed 224 \
  --obs-dtype float16 \
  --checkpoint-id 1
```

Weaker PPO, `epsilon = 0.05`:

```bash
python3 Y_Fu/collect_offline_data.py \
  --checkpoint Y_Fu/checkpoints/five_vs_five/update_10.pt \
  --policy checkpoint \
  --num-envs 6 \
  --total-env-steps 10000000 \
  --epsilon 0.05 \
  --chunk-size 500000 \
  --save-dir Y_Fu/offline_data/run_5v5_weaker_eps005 \
  --seed 225 \
  --obs-dtype float16 \
  --checkpoint-id 2
```

Random policy:

```bash
python3 Y_Fu/collect_offline_data.py \
  --policy random \
  --num-envs 6 \
  --total-env-steps 7500000 \
  --epsilon 0.0 \
  --chunk-size 500000 \
  --save-dir Y_Fu/offline_data/run_5v5_random \
  --seed 226 \
  --obs-dtype float16 \
  --checkpoint-id 3
```

### Full IQL training

```bash
python3 Y_Fu/train_iql.py \
  --dataset-dirs \
    Y_Fu/offline_data/run_5v5_best_eps0 \
    Y_Fu/offline_data/run_5v5_best_eps15 \
    Y_Fu/offline_data/run_5v5_weaker_eps005 \
    Y_Fu/offline_data/run_5v5_random \
  --reward-key reward \
  --device cuda \
  --save-dir Y_Fu/checkpoints/iql_5v5_iter0 \
  --batch-size 4096 \
  --chunk-reuse-batches 32 \
  --learning-rate 3e-4 \
  --gamma 0.993 \
  --expectile 0.7 \
  --temperature 3.0 \
  --tau 0.005 \
  --normalize-rewards \
  --total-gradient-steps 1000000 \
  --eval-interval 10000 \
  --save-interval 50000 \
  --eval-episodes 20 \
  --seed 323
```

Evaluate the best checkpoint:

```bash
python3 Y_Fu/evaluate_iql.py \
  --checkpoint Y_Fu/checkpoints/iql_5v5_iter0/best.pt \
  --episodes 20 \
  --deterministic \
  --device cuda \
  --seed 323
```

## 6. Next Iteration Note

The current collector supports:

- PPO checkpoints
- random policy

It does not yet support collecting directly from IQL checkpoints.

If you want an IQL-driven data collection stage later, add explicit IQL checkpoint loading to:

- [collect_offline_data.py](/home/yuhan/Codes/RL/Learning-to-Play-Football-Multi-Agent-Reinforcement-Learning/Y_Fu/collect_offline_data.py)

Before that extension, the correct workflow is:

- PPO / random collection
- IQL training
- IQL evaluation

## Notes

- `train_iql.py` now writes `training_metadata.json` with dataset stats and reward normalization scale.
- Reward normalization is applied in training, not baked into the stored dataset.

```bash
python3 Y_Fu/train_iql.py \
  --dataset-dirs \
    Y_Fu/offline_data/run_5v5_best_eps0 \
    Y_Fu/offline_data/run_5v5_best_eps15 \
    Y_Fu/offline_data/run_5v5_weaker_eps005 \
    Y_Fu/offline_data/run_5v5_random \
    Y_Fu/offline_data/run_5v5_iql_iter0_eps15 \
  --reward-key reward \
  --device cuda \
  --save-dir Y_Fu/checkpoints/iql_5v5_iter1 \
  --batch-size 4096 \
  --learning-rate 3e-4 \
  --gamma 0.993 \
  --expectile 0.7 \
  --temperature 3.0 \
  --tau 0.005 \
  --total-gradient-steps 1000000 \
  --eval-interval 10000 \
  --save-interval 50000 \
  --eval-episodes 20 \
  --seed 424
```

## 7. Fine-Tune PPO From The Best IQL Checkpoint

The PPO trainer now supports `--init-checkpoint` from IQL checkpoints.

Use this if the IQL policy:

- passes more cleanly
- creates shots more reliably
- looks better in videos than the current PPO line

Resume online `5_vs_5` PPO from IQL:

```bash
python3 Y_Fu/train.py \
  --preset five_vs_five \
  --device cpu \
  --total-timesteps 1000000 \
  --init-checkpoint Y_Fu/checkpoints/iql_5v5_iter0/best.pt
```

Evaluate the fine-tuned PPO checkpoint:

```bash
python3 Y_Fu/evaluate.py \
  --checkpoint Y_Fu/checkpoints/five_vs_five/latest.pt \
  --episodes 20 \
  --deterministic \
  --device cpu \
  --seed 523
```

## Notes

- Use Academy as PPO curriculum, not as an offline dataset mixed with `5_vs_5`.
- Keep offline RL focused on `5_vs_5` unless you explicitly build task-conditioned multi-stage offline training.
- Use `reward` first, not `score_reward`, because the collector now preserves the same shaping configuration as the source checkpoint.
- Do not assume `latest.pt` is best. Check metrics and video.
- Do not start full offline collection until the early `5_vs_5` transfer check looks real.
