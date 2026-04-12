# Phase 2 Smoke Tests

## What Was Tested

- syntax check for the edited Phase 2 files
- observation and centralized `share_obs` shapes for each structure variant
- end-to-end training loop execution for all `2` rewards x `3` structure variants
- checkpoint writing and metrics logging for all smoke runs

## Commands Run

```bash
python3 -m py_compile \
  Two_V_Two/train_basic.py \
  Two_V_Two/env/grf_simple_env.py \
  Two_V_Two/training/basic_shared_ppo.py \
  Two_V_Two/run_phase2_pilots.py \
  Two_V_Two/evaluation/run_r3_checkpoint_sweep.py

python3 - <<'PY'
from Two_V_Two.train_basic import build_parser
from Two_V_Two.env.grf_simple_env import TwoVTwoFootballEnv

variants = ['shared_ppo', 'shared_ppo_id', 'mappo_id_cc']
for structure in variants:
    args = build_parser().parse_args([
        '--disable_cuda',
        '--reward_variant', 'r3_assist',
        '--structure_variant', structure,
        '--episode_length', '64',
        '--n_rollout_threads', '1',
    ])
    env = TwoVTwoFootballEnv(args, rank=0, log_dir='Two_V_Two/results/phase2_smoke/shape_probe', is_eval=True)
    obs, share_obs, available_actions = env.reset()
    print(structure, obs.shape, share_obs.shape, available_actions.shape)
    env.close()
PY

python3 Two_V_Two/run_phase2_pilots.py \
  --disable_cuda \
  --num_env_steps 128 \
  --episode_length 64 \
  --n_rollout_threads 1 \
  --save_interval 1 \
  --output_root Two_V_Two/results/phase2_smoke
```

## Shape Checks

- `shared_ppo`
  - actor obs shape: `(2, 115)`
  - critic / share obs shape: `(2, 115)`
- `shared_ppo_id`
  - actor obs shape: `(2, 117)`
  - critic / share obs shape: `(2, 117)`
- `mappo_id_cc`
  - actor obs shape: `(2, 117)`
  - critic / share obs shape: `(2, 234)`

Interpretation:

- identity augmentation adds the expected `2` one-hot features
- centralized critic input is the concatenation of both agent observations

## Pass / Fail Status

- `shared_ppo`: passed
- `shared_ppo_id`: passed
- `mappo_id_cc`: passed

All six smoke runs completed:

- `r2_progress/shared_ppo`
- `r2_progress/shared_ppo_id`
- `r2_progress/mappo_id_cc`
- `r3_assist/shared_ppo`
- `r3_assist/shared_ppo_id`
- `r3_assist/mappo_id_cc`

## Fixes Needed During Smoke Work

- added an explicit project-root bootstrap in `run_r3_checkpoint_sweep.py`
- corrected the sweep summary writer to use the checkpoint field name actually produced by the evaluator

No additional Phase 2 code fixes were required after the smoke runs themselves.
