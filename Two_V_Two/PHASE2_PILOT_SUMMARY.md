# Phase 2 Pilot Summary

## Scope

Phase 2 implementation and pilot runs are complete for the planned 3 structure variants x 2 reward settings:

- rewards: `r2_progress`, `r3_assist`
- structures: `shared_ppo`, `shared_ppo_id`, `mappo_id_cc`

This step was limited to implementation, smoke validation, and modest pilot training. It did not start Phase 3 or any new reward search.

## Files Created / Modified

Created:

- `Two_V_Two/PHASE2_PLAN.md`
- `Two_V_Two/PHASE2_SMOKE_TESTS.md`
- `Two_V_Two/PHASE2_PILOT_SUMMARY.md`
- `Two_V_Two/run_phase2_pilots.py`

Modified:

- `Two_V_Two/train_basic.py`
- `Two_V_Two/env/grf_simple_env.py`
- `Two_V_Two/training/basic_shared_ppo.py`

## Structure Variants Implemented

### `shared_ppo`

- existing shared-policy PPO baseline
- local actor observation only
- no explicit identity
- no centralized critic

### `shared_ppo_id`

- same shared-policy PPO path as baseline
- appends a 2D one-hot identity to each agent observation:
  - agent 0: `[1, 0]`
  - agent 1: `[0, 1]`
- critic remains non-centralized

### `mappo_id_cc`

- shared actor uses local identity-augmented observations
- separate centralized critic uses joint information
- centralized critic input is the concatenation of both agents' actor observations, repeated per agent as `share_obs`

## Implementation Notes

Identity addition:

- implemented in `Two_V_Two/env/grf_simple_env.py`
- only applied for `shared_ppo_id` and `mappo_id_cc`
- observed actor input dimensions:
  - `shared_ppo`: `115`
  - `shared_ppo_id`: `117`
  - `mappo_id_cc`: actor `117`, critic `234`

Centralized critic:

- implemented in `Two_V_Two/training/basic_shared_ppo.py`
- baseline and identity-only variants keep the existing local-value path
- `mappo_id_cc` uses a dedicated centralized value network and critic optimizer
- PPO value targets for `mappo_id_cc` are computed from centralized `share_obs` only

Logging:

- preserved existing episode metrics
- added `structure_variant` into training metrics and console logging
- Phase 2 outputs are separated by reward and structure variant

## Smoke Tests

Smoke checks passed for all 6 conditions.

Commands run:

```bash
python3 -m py_compile \
  Two_V_Two/train_basic.py \
  Two_V_Two/env/grf_simple_env.py \
  Two_V_Two/training/basic_shared_ppo.py \
  Two_V_Two/run_phase2_pilots.py \
  Two_V_Two/evaluation/run_r3_checkpoint_sweep.py
```

```bash
python3 - <<'PY'
from Two_V_Two.train_basic import build_parser
from Two_V_Two.env.grf_simple_env import TwoVTwoFootballEnv

for structure in ['shared_ppo', 'shared_ppo_id', 'mappo_id_cc']:
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
```

```bash
python3 Two_V_Two/run_phase2_pilots.py \
  --disable_cuda \
  --num_env_steps 128 \
  --episode_length 64 \
  --n_rollout_threads 1 \
  --save_interval 1 \
  --output_root Two_V_Two/results/phase2_smoke
```

Observed smoke shapes:

- `shared_ppo`: `obs=(2, 115)`, `share_obs=(2, 115)`
- `shared_ppo_id`: `obs=(2, 117)`, `share_obs=(2, 117)`
- `mappo_id_cc`: `obs=(2, 117)`, `share_obs=(2, 234)`

All 6 smoke runs completed without crash.

## Pilot Runs Completed

Pilot budget:

- `200000` env steps
- `episode_length=400`
- `n_rollout_threads=4`

Command run:

```bash
python3 Two_V_Two/run_phase2_pilots.py \
  --disable_cuda \
  --num_env_steps 200000 \
  --episode_length 400 \
  --n_rollout_threads 4 \
  --save_interval 10 \
  --output_root Two_V_Two/results/phase2
```

Completed conditions:

- `r2_progress/shared_ppo`
- `r2_progress/shared_ppo_id`
- `r2_progress/mappo_id_cc`
- `r3_assist/shared_ppo`
- `r3_assist/shared_ppo_id`
- `r3_assist/mappo_id_cc`

## Saved Outputs

Smoke outputs:

- `Two_V_Two/results/phase2_smoke/`

Pilot outputs:

- `Two_V_Two/results/phase2/r2_progress/shared_ppo/`
- `Two_V_Two/results/phase2/r2_progress/shared_ppo_id/`
- `Two_V_Two/results/phase2/r2_progress/mappo_id_cc/`
- `Two_V_Two/results/phase2/r3_assist/shared_ppo/`
- `Two_V_Two/results/phase2/r3_assist/shared_ppo_id/`
- `Two_V_Two/results/phase2/r3_assist/mappo_id_cc/`

Each pilot directory contains `metrics.jsonl` and checkpoints.

## Pilot Metrics Snapshot

Final logged metrics at `200k`:

| Reward | Structure | Return | Goals | Passes | Pass-to-shot | Assists | Possession |
|---|---|---:|---:|---:|---:|---:|---:|
| R2 | `shared_ppo` | 0.0151 | 0.00 | 0.00 | 0.00 | 0.00 | 4.88 |
| R2 | `shared_ppo_id` | 0.0077 | 0.00 | 0.00 | 0.00 | 0.00 | 4.19 |
| R2 | `mappo_id_cc` | -0.0143 | 0.01 | 0.00 | 0.00 | 0.00 | 4.86 |
| R3 | `shared_ppo` | -0.0124 | 0.00 | 0.05 | 0.00 | 0.00 | 5.05 |
| R3 | `shared_ppo_id` | -0.0809 | 0.00 | 0.00 | 0.00 | 0.00 | 4.40 |
| R3 | `mappo_id_cc` | -0.0103 | 0.00 | 0.02 | 0.00 | 0.00 | 5.10 |

Peak pass-related pilot metrics during training:

| Reward | Structure | Peak pass step | Peak pass | Peak pass-to-shot step | Peak pass-to-shot |
|---|---|---:|---:|---:|---:|
| R2 | `shared_ppo` | 76800 | 0.00 | 76800 | 0.00 |
| R2 | `shared_ppo_id` | 16000 | 0.00 | 16000 | 0.00 |
| R2 | `mappo_id_cc` | 60800 | 0.00 | 60800 | 0.00 |
| R3 | `shared_ppo` | 147200 | 0.09 | 62400 | 0.03 |
| R3 | `shared_ppo_id` | 76800 | 0.09 | 35200 | 0.02 |
| R3 | `mappo_id_cc` | 4800 | 0.0741 | 11200 | 0.0441 |

## Early Interpretation

The pilot supports the intended ablation ladder:

1. `R2` remains a mostly non-cooperative control reward across all three structure settings.
2. `R3` continues to induce pass-level behavior that `R2` does not.
3. identity alone did not clearly stabilize that behavior in this pilot.
4. the centralized critic variant produced the strongest early pass-to-shot signal under `R3`, but the effect weakened by the final 200k checkpoint.

Practical reading of the pilot:

- `shared_ppo` remains the clean baseline.
- `shared_ppo_id` is implemented correctly, but did not show a clear pilot advantage.
- `mappo_id_cc` is the most promising Phase 2 target under `R3`, because it produced the clearest early joint-behavior signal while keeping the intervention limited to identity plus centralized value structure.

## Caveats

- These are pilot-training metrics, not a full deterministic checkpoint evaluation sweep.
- No pilot condition produced assists.
- The strongest `R3` differences appeared as transient or early pass / pass-to-shot structure, not yet stable end-state cooperation.
- A longer Phase 2 comparison and deterministic evaluation are still needed before making strong claims.

## Recommended Next Step

Proceed with focused Phase 2 comparison runs on:

- `R2/shared_ppo`
- `R2/mappo_id_cc`
- `R3/shared_ppo`
- `R3/mappo_id_cc`

`shared_ppo_id` is implemented and available, but based on this pilot it should remain a secondary ablation unless later deterministic evaluation shows a clearer benefit.
