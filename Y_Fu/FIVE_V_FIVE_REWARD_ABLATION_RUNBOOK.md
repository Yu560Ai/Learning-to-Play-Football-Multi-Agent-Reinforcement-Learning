# Five-v-Five Reward Ablation Runbook

## Purpose

This runbook is the operational companion to:

- [FIVE_V_FIVE_REWARD_V2_PLAN.md](/home/yuhan/Codes/RL/Learning-to-Play-Football-Multi-Agent-Reinforcement-Learning/Y_Fu/FIVE_V_FIVE_REWARD_V2_PLAN.md)
- [FIVE_V_FIVE_REWARD_V2_ABLATION_TABLE.md](/home/yuhan/Codes/RL/Learning-to-Play-Football-Multi-Agent-Reinforcement-Learning/Y_Fu/FIVE_V_FIVE_REWARD_V2_ABLATION_TABLE.md)

It answers:

1. in what order should the new reward variants be run?
2. when should a run be stopped early?
3. what exact evidence should be recorded for each variant?

This file is meant to be used while actually operating the machine.

## Main Rule

Do not run all variants blindly to long budgets.

Use short, comparable early-transfer runs first.

The correct question is:

- which reward variant creates the cleanest early football behavior?

not:

- which one consumed the most timesteps?

## Recommended Order

Run the variants in this order:

1. `five_vs_five_reward_v2`
2. `five_vs_five_reward_v2b_transition`
3. `five_vs_five_reward_v2c_progression`

Reason:

- `v2` is the cleanest new default
- `v2b_transition` only matters if `v2` looks too careless
- `v2c_progression` only matters if the team still cannot enter danger often enough

## Shared Run Setup

Use the same setup for all three:

- same init checkpoint
- same seed for the first comparison
- same `num_envs`
- same rollout settings
- same early budget

Recommended shared setup:

- init checkpoint: `Y_Fu/checkpoints/academy_pass_and_shoot_with_keeper/update_10.pt`
- `num_envs = 4`
- `rollout_steps = 256`
- `update_epochs = 4`
- `num_minibatches = 1`
- early budget: `2_000_000` agent steps

This keeps the comparison attributable.

## Commands

### Variant A: `v2`

```bash
python Y_Fu/train.py \
  --preset five_vs_five_reward_v2 \
  --device cpu \
  --num-envs 4 \
  --rollout-steps 256 \
  --update-epochs 4 \
  --num-minibatches 1 \
  --total-timesteps 2000000 \
  --init-checkpoint Y_Fu/checkpoints/academy_pass_and_shoot_with_keeper/update_10.pt \
  --seed 42
```

### Variant B: `v2b_transition`

```bash
python Y_Fu/train.py \
  --preset five_vs_five_reward_v2b_transition \
  --device cpu \
  --num-envs 4 \
  --rollout-steps 256 \
  --update-epochs 4 \
  --num-minibatches 1 \
  --total-timesteps 2000000 \
  --init-checkpoint Y_Fu/checkpoints/academy_pass_and_shoot_with_keeper/update_10.pt \
  --seed 42
```

### Variant C: `v2c_progression`

```bash
python Y_Fu/train.py \
  --preset five_vs_five_reward_v2c_progression \
  --device cpu \
  --num-envs 4 \
  --rollout-steps 256 \
  --update-epochs 4 \
  --num-minibatches 1 \
  --total-timesteps 2000000 \
  --init-checkpoint Y_Fu/checkpoints/academy_pass_and_shoot_with_keeper/update_10.pt \
  --seed 42
```

## Early-Stop Rule

Check the run early.

Do not wait until the full budget if the structure is already clearly bad.

### First Check Window

Check around:

- `250k ~ 500k` env steps
- roughly `1M ~ 2M` agent steps in `5_vs_5`

### Stop Immediately If Most Of These Hold

- `goals_for` still pinned near zero
- repeated `3001`-step matches
- `final_third_entries_ep` still very low
- `shot_attempt_events_ep` still very low
- `shot_per_final_third_entry` still near zero
- `right_bias` remains strongly elevated
- `top_actions` still dominated by directional movement and weak release patterns

This means the run is structurally bad, not merely unfinished.

## Promotion Rule

A variant should be promoted to a longer run only if it shows at least some of:

- more frequent final-third entries
- more shots from those entries
- at least occasional goals for
- cleaner video behavior than the earlier `five_vs_five` failures

If the run only improves shaped return, it should not be promoted.

## What To Record For Each Variant

For each reward variant, record:

1. command used
2. start and end timestamps
3. checkpoint chosen for evaluation
4. `avg_return`
5. `avg_score_reward`
6. `avg_goal_diff`
7. `win_rate`
8. `final_third_entries_ep`
9. `shot_attempt_events_ep`
10. `shot_per_final_third_entry`
11. `own_half_turnovers_ep`
12. `opponent_dangerous_possessions_ep`
13. one representative video path
14. a 2 to 4 line qualitative verdict

## Suggested Evaluation Step

After the early run ends, evaluate the chosen checkpoint:

```bash
python Y_Fu/evaluate.py \
  --checkpoint <variant_checkpoint> \
  --episodes 20 \
  --deterministic \
  --device cpu \
  --seed 123
```

Then save one representative video:

```bash
python Y_Fu/evaluate.py \
  --checkpoint <variant_checkpoint> \
  --episodes 1 \
  --deterministic \
  --device cpu \
  --seed 123 \
  --save-video \
  --video-dir <video_dir>
```

## Fast Triage Guide

### If `v2` Looks Best

- keep `v2` as the current main reward line
- only run `v2b` or `v2c` if you still have a specific unresolved weakness

### If `v2` Creates Danger But Gives Away Too Much

- run `v2b_transition`

This means:

- attack quality is coming alive
- but transition discipline is still too weak

### If `v2` Still Cannot Reach Dangerous Zones Often Enough

- run `v2c_progression`

This means:

- the line may still need slightly more reward for meaningful forward improvement
- but should still avoid generic possession reward

## Minimal Recording Template

Use this block in the stage log or a temporary note:

```text
Variant:
Command:
Checkpoint:
Episodes:
avg_return:
avg_score_reward:
avg_goal_diff:
win_rate:
final_third_entries_ep:
shot_attempt_events_ep:
shot_per_final_third_entry:
own_half_turnovers_ep:
opponent_dangerous_possessions_ep:
Video:
Verdict:
```

## Final Decision Rule

The winner is not the variant with the highest shaped return.

The winner is the variant that best improves:

1. goals and goal difference
2. attack completion quality
3. transition discipline
4. visible football structure

That is the only ranking rule that matters for the next PPO line.
