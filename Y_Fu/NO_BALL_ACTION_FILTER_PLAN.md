# No-Ball Action Filter Plan

## Problem

Recent `five_vs_five` videos exposed a specific degenerate behavior:

- a player may repeatedly choose `pass`-family actions even when that player does not own the ball
- this makes the policy look unstable and can trap training in a fake local optimum

This is not only a reward issue.

It is also an action-validity issue:

- the policy is allowed to emit ball-only actions from non-ball players
- those samples then pollute rollout statistics
- PPO can keep reinforcing a behavior that is not meaningful football control

## Fix

We now apply a light action filter inside the local environment wrapper.

Rule:

- if a controlled player does not own the ball
- and that player selects a ball-required action
- then the executed action is replaced with `idle`

Current ball-required actions:

- `long_pass`
- `high_pass`
- `short_pass`
- `shot`
- `dribble`

## PPO Alignment

Filtering at the environment boundary is not enough by itself.

If PPO still stores the originally sampled action, the batch becomes inconsistent with the actual transition.

So the trainer now also:

- reads back the `executed_actions` from env `info`
- stores executed actions in the rollout buffer
- recomputes old log-probabilities for those executed actions

This keeps PPO closer to the true behavior policy under the action filter.

## New Diagnostics

The rollout log should now expose:

- `invalid_ball_skill_rate`
- `invalid_no_ball_pass_rate`
- `invalid_no_ball_shot_rate`

These metrics are intended to answer:

- how often the raw policy still tries invalid ball-only actions
- whether the failure is shrinking or just being hidden by the filter

## Expected Effect

This fix should reduce one specific failure mode:

- "players keep passing without the ball"

It does **not** by itself guarantee good football.

It is a structural cleanup:

- remove invalid action spam
- make rollout data cleaner
- make behavior diagnostics more honest

## Decision Rule

If behavior improves after this fix:

- keep the filter
- continue evaluating `player_id` and reward settings on top of it

If the invalid-action rate drops but football is still poor:

- the next bottleneck is probably role formation or reward structure, not simple action validity
