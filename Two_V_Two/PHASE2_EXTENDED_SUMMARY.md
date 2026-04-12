# Phase 2 Extended Summary

## Scope

Focused Phase 2 extension completed for exactly these 4 conditions:

- `r2_progress/shared_ppo`
- `r2_progress/mappo_id_cc`
- `r3_assist/shared_ppo`
- `r3_assist/mappo_id_cc`

Budget:

- `500k` env steps per condition
- `20` deterministic eval episodes per saved checkpoint
- full saved-checkpoint sweep: `32` checkpoints per condition, `128` checkpoints total

## What Was Run

Training:

```bash
python3 Two_V_Two/run_phase2_extended.py \
  --disable_cuda \
  --num_env_steps 500000 \
  --episode_length 400 \
  --n_rollout_threads 4 \
  --save_interval 10 \
  --output_root Two_V_Two/results/phase2_extended
```

Deterministic checkpoint sweep:

```bash
python3 Two_V_Two/evaluation/run_phase2_checkpoint_sweep.py \
  --results_root Two_V_Two/results/phase2_extended \
  --episodes 20 \
  --checkpoint_stride 1 \
  --output_dir Two_V_Two/results/phase2_extended/analysis
```

Training-curve plot:

```bash
python3 Two_V_Two/evaluation/plot_phase2_extended.py \
  --results_root Two_V_Two/results/phase2_extended \
  --output_dir Two_V_Two/results/phase2_extended/analysis
```

## Main Artifacts

- training curves: `Two_V_Two/results/phase2_extended/analysis/training_curves.png`
- checkpoint eval table: `Two_V_Two/results/phase2_extended/analysis/checkpoint_eval_results.csv`
- checkpoint eval JSON: `Two_V_Two/results/phase2_extended/analysis/checkpoint_eval_results.json`
- best-checkpoint summary: `Two_V_Two/results/phase2_extended/analysis/best_checkpoints.md`
- ranking plot: `Two_V_Two/results/phase2_extended/analysis/deterministic_condition_ranking.png`
- sweep summary: `Two_V_Two/results/phase2_extended/analysis/summary.json`

## Training-Side Observation

Training traces still showed transient `R3` interaction structure:

- `r3_assist/shared_ppo`
  - peak training `pass_count = 0.06`
  - peak training `pass_to_shot_count = 0.02`
- `r3_assist/mappo_id_cc`
  - peak training `pass_count ~= 0.026`
  - peak training `pass_to_shot_count = 0.02`

So the training logs continued to suggest intermittent teammate-aware structure under `R3`.

## Deterministic Checkpoint Result

The deterministic sweep did **not** confirm that training-side signal.

Across all `128` evaluated checkpoints:

- `mean_pass_count = 0` for every checkpoint
- `mean_pass_to_shot_count = 0` for every checkpoint
- `mean_assist_count = 0` for every checkpoint

This held for both:

- `r3_assist/shared_ppo`
- `r3_assist/mappo_id_cc`

The only notable deterministic nonzero outcome in the full sweep was a small goal checkpoint under the control condition:

- `r2_progress/mappo_id_cc`
  - `update_000130.pt`
  - `208000` env steps
  - `goal_rate = 0.05`
  - still `pass_count = 0`
  - still `pass_to_shot_count = 0`

## Answers to the Phase 2 Questions

### 1. Does `r2_progress + mappo_id_cc` remain non-cooperative?

Yes.

It showed some isolated goal signal, but deterministic checkpoint eval stayed at:

- zero passes
- zero pass-to-shot
- zero assists

So multi-agent structure alone did not create teammate interaction under `R2`.

### 2. Does `r3_assist + shared_ppo` improve under longer budget?

Only in the training trace, not in deterministic checkpoint eval.

The training metrics still showed intermittent pass and pass-to-shot behavior, but the deterministic sweep found no reproducible pass-level separation at saved checkpoints.

### 3. Does `r3_assist + mappo_id_cc` show stronger or more stable deterministic pass-to-shot than `r3_assist + shared_ppo`?

No.

Under deterministic checkpoint eval, both stayed at zero for:

- `pass_count`
- `pass_to_shot_count`
- `assist_count`

So the longer-budget extension did not establish a deterministic advantage for `mappo_id_cc` over the shared baseline.

### 4. Did any condition show deterministic assist or consistent goal-related cooperative improvement?

No deterministic assist appeared anywhere.

There was no deterministic pass-based cooperative improvement in any condition. Goal-related deterministic signal was limited to isolated control checkpoints and was not accompanied by teammate interaction.

## Interpretation

The focused extension clarified the mismatch between:

- noisy or transient training-side interaction metrics
- reproducible deterministic checkpoint behavior

At this stage, the deterministic evidence is stronger than the training trace for decision-making.

The current Phase 2 setup does **not** support the claim that either:

- `R3/shared_ppo`, or
- `R3/mappo_id_cc`

produces stable deterministic cooperative structure at `500k`.

## Recommended Next Action

### Recommendation: 3. revise structure or reward if the signal disappears

Reason:

- the deterministic sweep did not recover pass-level or pass-to-shot separation for either `R3` condition
- `mappo_id_cc` did not turn the transient `R3` training signal into a reproducible checkpoint-level effect
- extending the same setup to a larger budget is not justified yet by the current evidence

Practical reading:

- keep the current Phase 2 artifacts as a negative result with useful controls
- do not expand the budget matrix immediately
- revisit the reward / eval interface or the interaction signal before the next architecture comparison
