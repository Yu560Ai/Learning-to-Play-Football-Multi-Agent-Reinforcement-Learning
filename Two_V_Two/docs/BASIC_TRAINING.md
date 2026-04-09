# Basic Training

## Entry Point

Run the baseline trainer from the repo root:

```bash
python3 Two_V_Two/train_basic.py
```

## Default Baseline

The current default setup is:

- scenario: `academy_run_pass_and_shoot_with_keeper`
- controlled players: `2`
- observation representation: `simple115v2`
- reward: `scoring,checkpoints`
- policy: shared-policy PPO
- value function: policy value head on the shared actor network

## Useful Overrides

Example longer run:

```bash
python3 Two_V_Two/train_basic.py \
  --disable_cuda \
  --num_env_steps 200000 \
  --n_rollout_threads 4 \
  --run_dir Two_V_Two/runs/basic_shared_ppo_v1
```

If you want the sparse baseline instead of checkpoints:

```bash
python3 Two_V_Two/train_basic.py --rewards scoring
```

## Outputs

Each run writes:

- config: `RUN_DIR/config.json`
- training metrics: `RUN_DIR/metrics.jsonl`
- checkpoints: `RUN_DIR/checkpoints/latest.pt`

## Smoke Test

Verified locally with:

```bash
python3 Two_V_Two/train_basic.py \
  --disable_cuda \
  --num_env_steps 64 \
  --episode_length 16 \
  --save_interval 1 \
  --run_dir Two_V_Two/runs/smoke_basic_shared_ppo
```
