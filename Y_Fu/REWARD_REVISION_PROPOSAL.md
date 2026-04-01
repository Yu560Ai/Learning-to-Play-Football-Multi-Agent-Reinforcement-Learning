# Reward Revision Proposal

## Purpose

This note proposes a concrete reward revision for the current `Y_Fu` PPO football setup.

The goal is:

- reduce rewards that are easy to exploit without solving the task
- reduce noisy team-level blame signals
- strengthen dense events that are closer to real scoring chances

This proposal is based on the current shaping logic in:

- [envs.py](/home/yuhan/Codes/RL/Learning-to-Play-Football-Multi-Agent-Reinforcement-Learning/Y_Fu/yfu_football/envs.py)
- [ppo.py](/home/yuhan/Codes/RL/Learning-to-Play-Football-Multi-Agent-Reinforcement-Learning/Y_Fu/yfu_football/ppo.py)

## Current Trigger Logic

The current shaping terms are triggered by:

- pass action followed by teammate possession
- shot action while owning the ball
- possession in advanced x range
- crossing the attacking x threshold
- possession retained between two steps
- possession recovery
- possession loss in own half
- opponent possession in a defensive-danger zone

This matters because some of these events are:

- near-causal to scoring
- frequent but weakly useful
- noisy and weakly attributable

## Main Diagnosis

The current setup likely over-rewards:

- successful passing as an end in itself
- generic possession
- safe retention

and under-rewards:

- action sequences that actually end in dangerous attacks

At the same time, some penalties are probably too noisy:

- generic team-level opponent possession in a dangerous area
- ball-loss penalties that may not be cleanly attributable to the trained player group

## Revision Rule

### Keep As Anchors

- GRF `scoring`
- GRF `checkpoints`

These should remain the base task direction.

### Reduce Or Remove

- generic possession reward
- generic possession retention reward
- heavy pass-completion reward if it promotes harmless cycling
- noisy opponent-side blame penalties

### Strengthen

- shot creation signals
- entering genuinely dangerous attacking zones
- forward progression that is attached to attacking improvement, not just possession duration

## Proposed Changes

## 1. `academy_pass_and_shoot_with_keeper`

Current values:

- `pass_success_reward = 0.08`
- `pass_failure_penalty = 0.04`
- `pass_progress_reward_scale = 0.08`
- `shot_attempt_reward = 0.03`
- `attacking_possession_reward = 0.002`
- `final_third_entry_reward = 0.04`
- `possession_retention_reward = 0.001`
- `possession_recovery_reward = 0.02`
- `defensive_third_recovery_reward = 0.03`
- `opponent_attacking_possession_penalty = 0.0015`
- `own_half_turnover_penalty = 0.02`

Proposed values:

- `pass_success_reward = 0.04`
- `pass_failure_penalty = 0.02`
- `pass_progress_reward_scale = 0.05`
- `shot_attempt_reward = 0.08`
- `attacking_possession_reward = 0.0`
- `final_third_entry_reward = 0.08`
- `possession_retention_reward = 0.0`
- `possession_recovery_reward = 0.01`
- `defensive_third_recovery_reward = 0.01`
- `opponent_attacking_possession_penalty = 0.0`
- `own_half_turnover_penalty = 0.01`

Why:

- passing should help create the shot, not become the substitute objective
- shot and dangerous entry should matter more than generic ball security
- defensive penalties should stay weak in this offensive curriculum task

## 2. `academy_3_vs_1_with_keeper`

Current values:

- `pass_success_reward = 0.08`
- `pass_failure_penalty = 0.05`
- `pass_progress_reward_scale = 0.08`
- `shot_attempt_reward = 0.03`
- `attacking_possession_reward = 0.002`
- `final_third_entry_reward = 0.04`
- `possession_retention_reward = 0.001`
- `possession_recovery_reward = 0.02`
- `defensive_third_recovery_reward = 0.03`
- `opponent_attacking_possession_penalty = 0.0015`
- `own_half_turnover_penalty = 0.02`

Proposed values:

- `pass_success_reward = 0.04`
- `pass_failure_penalty = 0.025`
- `pass_progress_reward_scale = 0.05`
- `shot_attempt_reward = 0.08`
- `attacking_possession_reward = 0.0`
- `final_third_entry_reward = 0.07`
- `possession_retention_reward = 0.0`
- `possession_recovery_reward = 0.01`
- `defensive_third_recovery_reward = 0.01`
- `opponent_attacking_possession_penalty = 0.0`
- `own_half_turnover_penalty = 0.01`

Why:

- same logic as `academy_pass_and_shoot_with_keeper`
- in `3_vs_1`, the model should be pushed to convert local advantage into a finish
- generic retention is especially risky here because it can reward "keep the ball moving" without actually scoring

## 3. `five_vs_five`

Current values:

- `pass_success_reward = 0.06`
- `pass_failure_penalty = 0.05`
- `pass_progress_reward_scale = 0.06`
- `shot_attempt_reward = 0.02`
- `attacking_possession_reward = 0.001`
- `final_third_entry_reward = 0.03`
- `possession_retention_reward = 0.0005`
- `possession_recovery_reward = 0.015`
- `defensive_third_recovery_reward = 0.025`
- `opponent_attacking_possession_penalty = 0.001`
- `own_half_turnover_penalty = 0.03`

Proposed values:

- `pass_success_reward = 0.03`
- `pass_failure_penalty = 0.025`
- `pass_progress_reward_scale = 0.04`
- `shot_attempt_reward = 0.06`
- `attacking_possession_reward = 0.0`
- `final_third_entry_reward = 0.06`
- `possession_retention_reward = 0.0`
- `possession_recovery_reward = 0.01`
- `defensive_third_recovery_reward = 0.02`
- `opponent_attacking_possession_penalty = 0.0`
- `own_half_turnover_penalty = 0.015`

Why:

- `five_vs_five` should reward attacking completion more strongly
- the current shot reward is likely too weak relative to pass and progression signals
- small generic retention rewards add up very often and can bias toward safe but empty play
- own-half turnover penalties should still exist, but be lighter because team-level football is noisy

## Terms To Be Most Suspicious Of

The highest-risk shaping terms in the current code are:

1. `attacking_possession_reward`

Reason:

- very frequent
- weakly tied to actual task completion
- can reward standing in a "good" zone without creating a finish

Recommendation:

- set to `0.0`

2. `possession_retention_reward`

Reason:

- extremely frequent
- tends to reward safe continuation rather than useful continuation

Recommendation:

- set to `0.0`

3. `opponent_attacking_possession_penalty`

Reason:

- noisy
- often reflects a larger team transition rather than a specific attributable mistake

Recommendation:

- set to `0.0` for now

## Terms To Make More Important

The best current candidate to strengthen is:

1. `shot_attempt_reward`

Reason:

- much closer to actual scoring than generic possession
- still denser than goals

2. `final_third_entry_reward`

Reason:

- also closer to real danger than pass completion alone
- works well as a bridge between checkpoint progress and shot creation

## One Important Interaction

The environment already uses:

- `rewards = "scoring,checkpoints"`

So the system already has a built-in progression signal.

That means extra shaping for:

- pass progress
- attacking possession
- retention

should be used carefully, because these can duplicate or overpower the intended GRF checkpoint structure.

## Is Reward Tuning "Algorithmic"?

Partly, but not necessarily in the meta-learning sense.

### In the current project

What we are doing is standard reward engineering:

- inspect failure behavior
- inspect which rewards are frequent
- inspect which rewards are causally useful
- revise the reward weights
- rerun evaluation

This is still algorithmic in a broad engineering sense, but it is not meta-learning.

### What meta-learning would mean

That would be something like:

- learning reward weights automatically
- using outer-loop optimization to tune reward coefficients
- meta-gradient RL
- population-based tuning

Those methods exist, but they are not the standard first move here.

### Standard approach for this project

The standard approach is:

1. use task-informed dense shaping
2. ablate and simplify shaping terms
3. validate by actual task metrics, not shaped return alone

That is the right next step for this repo.

## Recommended Next Action

If the current `five_vs_five` half-day checkpoint is weak, the next reward edit should be:

1. zero out:

- `attacking_possession_reward`
- `possession_retention_reward`
- `opponent_attacking_possession_penalty`

2. reduce:

- `pass_success_reward`
- `pass_failure_penalty`
- `own_half_turnover_penalty`

3. increase:

- `shot_attempt_reward`
- `final_third_entry_reward`

That is the smallest high-signal reward revision for the current codebase.
