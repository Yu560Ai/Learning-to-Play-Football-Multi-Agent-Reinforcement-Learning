# Phase 1 Extension Summary

## Scope Completed

- Completed 1M shared-policy PPO runs for `r2_progress`, `r3_assist`, and `r4_anti_selfish`.
- Completed 5M shared-policy PPO runs for `r2_progress` and `r3_assist`.
- Ran deterministic evaluation at 200k and 1M with the same eval script.
- Ran deterministic evaluation at 5M with the same eval script.
- Generated cross-budget training-curve plots and budget comparison plots.
- No Phase 2 changes were made.

## Pre-1M Expectation

- `R1`: remain sparse and mostly barren.
- `R2`: strongest early stable baseline; likely first useful dense-learning signal.
- `R3`: later activation than `R2`; expected to separate on pass or pass-to-shot behavior before assist becomes common.
- `R4`: strongest behavior-shaping pressure on possession and passing, but highest risk of becoming penalty-dominated.

## What Happened

### 200k reference

- `R1` stayed barren: no goals, no passes, no assists.
- `R2` was the cleanest dense baseline, but still no passing structure.
- `R3` showed only brief weak pass activity in training and no deterministic eval gains.
- `R4` produced the earliest pass signal in training, but it was noisy and already showed penalty sensitivity.

### 1M extension

- `R2` remained the most stable baseline in training. Training goal proxy rose slightly by the end (`mean_goal_count` tail about `0.02`), and deterministic eval return improved modestly vs 200k, but eval still showed zero goals and zero passes.
- `R3` produced intermittent nonzero `pass_count` and `pass_to_shot_count` during training, and at 1M it was the only deterministic eval checkpoint with a nonzero goal rate:
  - `goal_rate = 0.08`
  - `mean_goal_count = 0.08`
  - `mean_episode_return = 0.095`
- `R4` showed the strongest anti-selfish effect in training, with lower possession streaks and some pass / pass-to-shot bursts, but deterministic eval remained strongly negative and barren:
  - `mean_episode_return = -7.695`
  - `goal_rate = 0.0`
  - `mean_pass_count = 0.0`

## Comparison Against Expectation

- `R1`: matched expectation. It is still only a sparse reference baseline.
- `R2`: mostly matched expectation on stability, but did not convert the dense shaping into deterministic goal production by 1M.
- `R3`: matched the delayed-activation hypothesis better than the other cooperative variants. Assist events are still absent, but `R3` did show meaningful behavioral separation from `R2`:
  - nonzero training `pass_count`
  - nonzero training `pass_to_shot_count`
  - best deterministic eval goal rate and return at 1M
- `R4`: partially matched expectation. It changed behavior, but by 1M the evidence points to over-penalization / instability rather than a robust cooperative improvement.

## 5M Extension

### R2 at 5M

- `R2` remained exactly what Phase 1 suggested it was: a stable optimization baseline with no teammate interaction.
- Final deterministic 5M eval:
  - `goal_rate = 0.0`
  - `mean_goal_count = 0.0`
  - `mean_pass_count = 0.0`
  - `mean_assist_count = 0.0`
  - `mean_pass_to_shot_count = 0.0`

### R3 at 5M

- `R3` did show delayed cooperative structure during training, but only intermittently.
- Across the 5M run, training repeatedly produced nonzero pass-level behavior, including windows around:
  - `mean_pass_count ~ 0.03-0.05`
  - `mean_pass_to_shot_count ~ 0.01-0.02`
- Those windows were not stable. They repeatedly faded back to near-zero interaction, and `assist_count` never became nonzero.
- Final deterministic 5M eval also collapsed to a barren endpoint:
  - `goal_rate = 0.0`
  - `mean_goal_count = 0.0`
  - `mean_pass_count = 0.0`
  - `mean_assist_count = 0.0`
  - `mean_pass_to_shot_count = 0.0`

## Final Long-Budget Judgment

### R2

- The long-budget result is clean: `R2` is a useful stable non-cooperative baseline.
- It should be kept as the baseline reference for future comparisons, but not treated as evidence of cooperation.

### R3

- The long-budget result is mixed.
- `R3` does show delayed activation of teammate-aware attack structure in training, because it repeatedly separates from `R2` on `pass_count` and `pass_to_shot_count`.
- But that structure is not yet robust enough to survive into the final deterministic 5M checkpoint.
- So the correct scientific reading is:
  - `R3` is not behaving just like `R2` throughout training.
  - `R3` does show real but unstable cooperative shaping.
  - `R3` has **not** yet produced a solid long-budget cooperative policy under the current Phase 1 setup.

## 5M Continuation Decision

Decision:

- Continue to 5M: `R2`, `R3`
- Do not continue to 5M: `R4`
- Keep as reference only: `R1`

Justification:

- `R2` qualifies as the stable baseline to carry forward. It remains the cleanest optimization reference and does show some training-side improvement from 200k to 1M, even though deterministic goal production is still weak.
- `R3` qualifies because it shows meaningful behavioral separation from `R2` by 1M. The key evidence is not assist count yet; it is the combination of training-time pass / pass-to-shot signal plus the best deterministic eval goal rate and return.
- `R4` does **not** qualify for 5M. Its training curves show behavior shaping, but the 1M deterministic eval is strongly penalty-dominated and does not preserve the intended cooperative behavior well enough to justify a larger budget.

Post-5M recommendation:

- `R2` should remain the baseline carried forward.
- `R3` should only be carried forward with caution. It is scientifically interesting because of the repeated delayed pass-level activations, but it is not yet a solid cooperative reward winner.
- On the current evidence, the safer next step is still inside Phase 1: keep `R2` as baseline and either refine `R3` reward / checkpoint selection / evaluation before Phase 2, or enter Phase 2 knowing that `R3` is exploratory rather than validated.

## Commands Run

```bash
python3 Two_V_Two/run_phase1_variants.py \
  --disable_cuda \
  --variants r2_progress r3_assist r4_anti_selfish \
  --n_rollout_threads 4 \
  --episode_length 400 \
  --num_env_steps 1000000 \
  --save_interval 25 \
  --output_root Two_V_Two/results/phase1_extended/budget_1000000

python3 Two_V_Two/evaluation/run_phase1_eval.py \
  --results_root Two_V_Two/results/phase1_200k \
  --episodes 50 \
  --output_json Two_V_Two/results/phase1_extended/budget_200k/eval_summary.json

python3 Two_V_Two/evaluation/run_phase1_eval.py \
  --results_root Two_V_Two/results/phase1_extended/budget_1000000 \
  --variants r2_progress r3_assist r4_anti_selfish \
  --episodes 50 \
  --output_json Two_V_Two/results/phase1_extended/budget_1000000/eval_summary.json

python3 Two_V_Two/evaluation/plot_phase1_extended.py \
  --budget_root 200k=Two_V_Two/results/phase1_200k \
  --budget_root 1M=Two_V_Two/results/phase1_extended/budget_1000000 \
  --output_dir Two_V_Two/results/phase1_extended/plots \
  --window 5

python3 Two_V_Two/run_phase1_variants.py \
  --disable_cuda \
  --variants r2_progress r3_assist \
  --n_rollout_threads 4 \
  --episode_length 400 \
  --num_env_steps 5000000 \
  --save_interval 50 \
  --output_root Two_V_Two/results/phase1_extended/budget_5000000

python3 Two_V_Two/evaluation/run_phase1_eval.py \
  --results_root Two_V_Two/results/phase1_extended/budget_5000000 \
  --variants r2_progress r3_assist \
  --episodes 50 \
  --output_json Two_V_Two/results/phase1_extended/budget_5000000/eval_summary.json

python3 Two_V_Two/evaluation/plot_phase1_extended.py \
  --budget_root 200k=Two_V_Two/results/phase1_200k \
  --budget_root 1M=Two_V_Two/results/phase1_extended/budget_1000000 \
  --budget_root 5M=Two_V_Two/results/phase1_extended/budget_5000000 \
  --output_dir Two_V_Two/results/phase1_extended/plots \
  --window 5
```

## Output Locations

- 200k training logs: `Two_V_Two/results/phase1_200k/`
- 1M training logs: `Two_V_Two/results/phase1_extended/budget_1000000/`
- 5M training logs: `Two_V_Two/results/phase1_extended/budget_5000000/`
- 200k eval summary: `Two_V_Two/results/phase1_extended/budget_200k/eval_summary.json`
- 1M eval summary: `Two_V_Two/results/phase1_extended/budget_1000000/eval_summary.json`
- 5M eval summary: `Two_V_Two/results/phase1_extended/budget_5000000/eval_summary.json`
- plots: `Two_V_Two/results/phase1_extended/plots/`

## Caveats

- Assist counts are still zero at both budgets. Phase 1 has not yet produced clear assisted-goal behavior.
- Deterministic eval is harsher than the training traces; some training-time pass signals collapse under greedy action selection.
- `R2` remains useful mainly as the stable baseline, not as evidence of cooperation by itself.
- The 1M `R3` checkpoint looked stronger than the final 5M `R3` checkpoint. That means "latest checkpoint" is not necessarily the best checkpoint for cooperative behavior.
