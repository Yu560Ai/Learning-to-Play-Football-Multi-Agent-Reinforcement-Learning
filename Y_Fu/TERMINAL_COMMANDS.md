# Y_Fu Terminal Commands

## Start Here

From the repository root:

```bash
cd ~/Codes/RL/Learning-to-Play-Football-Multi-Agent-Reinforcement-Learning
source football-master/football-env/bin/activate
```

## Main Training Line

Use this order:

1. Academy PPO
2. `5_vs_5` PPO
3. `5_vs_5` offline RL
4. PPO fine-tuning from IQL if IQL is better

## Academy PPO

### `academy_run_to_score_with_keeper`

```bash
python -u Y_Fu/train.py --preset academy_run_to_score_with_keeper --device cpu
```

### `academy_pass_and_shoot_with_keeper`

```bash
python -u Y_Fu/train.py --preset academy_pass_and_shoot_with_keeper --device cpu
```

Evaluate:

```bash
python Y_Fu/evaluate.py --checkpoint Y_Fu/checkpoints/academy_pass_and_shoot_with_keeper/latest.pt --episodes 20 --deterministic --compare-random --device cpu --seed 123
```

### `academy_3_vs_1_with_keeper`

Start from the best `academy_pass_and_shoot_with_keeper` checkpoint:

```bash
python -u Y_Fu/train.py --preset academy_3_vs_1_with_keeper --device cpu --init-checkpoint Y_Fu/checkpoints/academy_pass_and_shoot_with_keeper/latest.pt
```

Continue from an existing `academy_3_vs_1_with_keeper` checkpoint:

```bash
python -u Y_Fu/train.py --preset academy_3_vs_1_with_keeper --device cpu --total-timesteps 300000 --init-checkpoint Y_Fu/checkpoints/academy_3_vs_1_with_keeper/latest.pt
```

Evaluate:

```bash
python Y_Fu/evaluate.py --checkpoint Y_Fu/checkpoints/academy_3_vs_1_with_keeper/latest.pt --episodes 20 --deterministic --compare-random --device cpu --seed 123
```

Render one episode:

```bash
python Y_Fu/evaluate.py --checkpoint Y_Fu/checkpoints/academy_3_vs_1_with_keeper/latest.pt --episodes 1 --deterministic --render --device cpu
```

## `5_vs_5` PPO

Start from the best Academy checkpoint:

```bash
python -u Y_Fu/train.py --preset five_vs_five --device cpu --total-timesteps 1000000 --init-checkpoint Y_Fu/checkpoints/academy_3_vs_1_with_keeper/latest.pt
```

Start from a better earlier Academy checkpoint if that one looks cleaner:

```bash
python -u Y_Fu/train.py --preset five_vs_five --device cpu --total-timesteps 1000000 --init-checkpoint Y_Fu/checkpoints/academy_3_vs_1_with_keeper/update_90.pt
```

Continue a `5_vs_5` PPO run from an existing checkpoint:

```bash
python -u Y_Fu/train.py --preset five_vs_five --device cpu --total-timesteps 1000000 --init-checkpoint Y_Fu/checkpoints/five_vs_five/update_70.pt
```

Evaluate the current `5_vs_5` checkpoint:

```bash
python Y_Fu/evaluate.py --checkpoint Y_Fu/checkpoints/five_vs_five/latest.pt --episodes 20 --deterministic --compare-random --device cpu --seed 123
```

Render one episode:

```bash
python Y_Fu/evaluate.py --checkpoint Y_Fu/checkpoints/five_vs_five/latest.pt --episodes 1 --deterministic --render --device cpu
```

## `5_vs_5` Offline RL

Pilot collection from the best PPO checkpoint:

```bash
python3 Y_Fu/collect_offline_data.py --checkpoint Y_Fu/checkpoints/five_vs_five/latest.pt --policy checkpoint --num-envs 6 --total-env-steps 200000 --epsilon 0.0 --chunk-size 500000 --save-dir Y_Fu/offline_data/pilot_5v5_best_eps0 --seed 123 --obs-dtype float16 --checkpoint-id 1
```

Pilot collection with exploration:

```bash
python3 Y_Fu/collect_offline_data.py --checkpoint Y_Fu/checkpoints/five_vs_five/latest.pt --policy checkpoint --num-envs 6 --total-env-steps 100000 --epsilon 0.15 --chunk-size 500000 --save-dir Y_Fu/offline_data/pilot_5v5_best_eps15 --seed 124 --obs-dtype float16 --checkpoint-id 1
```

Pilot collection from a weaker PPO checkpoint:

```bash
python3 Y_Fu/collect_offline_data.py --checkpoint Y_Fu/checkpoints/five_vs_five/update_10.pt --policy checkpoint --num-envs 6 --total-env-steps 100000 --epsilon 0.05 --chunk-size 500000 --save-dir Y_Fu/offline_data/pilot_5v5_weaker_eps005 --seed 125 --obs-dtype float16 --checkpoint-id 2
```

Pilot collection from a random policy:

```bash
python3 Y_Fu/collect_offline_data.py --policy random --num-envs 6 --total-env-steps 100000 --epsilon 0.0 --chunk-size 500000 --save-dir Y_Fu/offline_data/pilot_5v5_random --seed 126 --obs-dtype float16 --checkpoint-id 3
```

Pilot IQL training:

```bash
python3 Y_Fu/train_iql.py --dataset-dirs Y_Fu/offline_data/pilot_5v5_best_eps0 Y_Fu/offline_data/pilot_5v5_best_eps15 Y_Fu/offline_data/pilot_5v5_weaker_eps005 Y_Fu/offline_data/pilot_5v5_random --reward-key reward --device cuda --save-dir Y_Fu/checkpoints/iql_5v5_pilot --batch-size 4096 --learning-rate 3e-4 --gamma 0.993 --expectile 0.7 --temperature 3.0 --tau 0.005 --total-gradient-steps 20000 --eval-interval 5000 --save-interval 10000 --eval-episodes 20 --seed 123
```

Evaluate the pilot IQL checkpoint:

```bash
python3 Y_Fu/evaluate_iql.py --checkpoint Y_Fu/checkpoints/iql_5v5_pilot/best.pt --episodes 20 --deterministic --device cuda --seed 123
```

## PPO Fine-Tuning From IQL

The PPO trainer now supports `--init-checkpoint` from IQL checkpoints.

Resume `5_vs_5` PPO from the best IQL checkpoint:

```bash
python -u Y_Fu/train.py --preset five_vs_five --device cpu --total-timesteps 1000000 --init-checkpoint Y_Fu/checkpoints/iql_5v5_iter0/best.pt
```

Evaluate the fine-tuned PPO checkpoint:

```bash
python Y_Fu/evaluate.py --checkpoint Y_Fu/checkpoints/five_vs_five/latest.pt --episodes 20 --deterministic --compare-random --device cpu --seed 523
```

## SaltyFish-Inspired Baseline

Train the separate single-player competition-style baseline:

```bash
python Y_Fu/train_saltyfish.py --device cpu
```

Continue from an older checkpoint:

```bash
python Y_Fu/train_saltyfish.py --device cpu --init-checkpoint Y_Fu/checkpoints/saltyfish_baseline/latest.pt
```

Run a shorter smoke test:

```bash
python Y_Fu/train_saltyfish.py --device cpu --total-timesteps 50000 --rollout-steps 256
```

Evaluate:

```bash
python Y_Fu/evaluate_saltyfish.py --checkpoint Y_Fu/checkpoints/saltyfish_baseline/latest.pt --episodes 5 --compare-random --device cpu
```

## Save Video

Save one `academy_3_vs_1_with_keeper` video:

```bash
python Y_Fu/evaluate.py --checkpoint Y_Fu/checkpoints/academy_3_vs_1_with_keeper/latest.pt --episodes 1 --deterministic --device cpu --save-video --video-dir Y_Fu/videos/academy_3_vs_1_with_keeper
```

Save one `5_vs_5` video:

```bash
python Y_Fu/evaluate.py --checkpoint Y_Fu/checkpoints/five_vs_five/latest.pt --episodes 1 --deterministic --device cpu --save-video --video-dir Y_Fu/videos/five_vs_five_latest
```

## Notes

- Do not mix Academy and `5_vs_5` datasets in the same initial IQL run.
- Use preset-specific checkpoints, not a generic `latest.pt` from another stage.
- `latest.pt` is not automatically the best checkpoint.
- Read [OFFLINE_RL_COMMANDS.md](/home/yuhan/Codes/RL/Learning-to-Play-Football-Multi-Agent-Reinforcement-Learning/Y_Fu/OFFLINE_RL_COMMANDS.md) for the full hybrid command flow.
- Read [PPO_IQL_EXECUTION_CHECKLIST.md](/home/yuhan/Codes/RL/Learning-to-Play-Football-Multi-Agent-Reinforcement-Learning/Y_Fu/PPO_IQL_EXECUTION_CHECKLIST.md) for the shortest execution checklist.
