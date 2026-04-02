# Academy Reboot Plan

## Why Pause The Current `five_vs_five` PPO Line

The current `five_vs_five` PPO exploration has already produced enough evidence:

- long runs can stay stuck in poor football behavior
- reward-only revision was not enough
- `player_id` alone was not enough
- action-validity cleanup was necessary, but should be tested in a simpler setting first

So the current decision is:

- pause the main `five_vs_five` PPO exploration
- return to Academy
- use Academy as a controlled primitive-learning and behavior-diagnosis stage

This is not a retreat from the final goal.

It is a reset toward a more controllable training loop.

## Final Goal Does Not Change

The final target remains:

- `five_vs_five`
- 4 learned outfield agents
- goalkeeper controlled by the environment

Academy is only the bootstrap and diagnosis layer.

## Immediate Reboot Strategy

Use a fresh Academy line with:

- the new no-ball action filter
- explicit `player_id` for multi-player Academy stages
- frequent checkpoints
- strict pass/fail gates

Do not let Academy absorb unlimited budget.

## New Active Order

### Step 1. Controlled passing reboot

Start with:

- `academy_pass_and_shoot_with_keeper`

Why first:

- it is the most directly relevant controlled task for the current failure
- it tests passing, receiving, and finishing
- it is the easiest place to verify that no-ball pass spam is actually reduced

Recommended run shape:

- start from scratch
- `use_player_id=True`
- `num_envs=4`
- moderate rollout size
- frequent checkpointing

Primary metrics:

- scored-episode ratio
- `avg_score_reward`
- `pass_rate`
- `shot_rate`
- `invalid_no_ball_pass_rate`

Pass condition:

- the policy must show real pass-then-shoot behavior
- not just moving and not just invalid pass spam

### Step 2. Direct carry-and-finish sanity check

Use:

- `academy_run_to_score_with_keeper`

Purpose:

- verify that direct attack and finishing primitives are not fundamentally broken

This stage is simpler than the passing task and should help separate:

- "passing logic is broken"
from
- "basic attack geometry is broken"

### Step 3. Optional `academy_3_vs_1_with_keeper`

Only move here if Step 1 and Step 2 look alive.

Purpose:

- test one more layer of support behavior
- test whether simple coordination generalizes beyond two-player pass-and-finish

If the reboot still fails before this point, do not expand the task again.

## Hard Rules

### Rule 1. Fresh start

Do not warm-start from the failed `five_vs_five` checkpoints.

The reboot should not inherit clearly bad `five_vs_five` behavior.

### Rule 2. One active run

Keep one main training run only.

Do not reopen multiple long PPO runs in parallel.

### Rule 3. Checkpoint-driven evaluation

At named checkpoints:

- run short deterministic evaluation
- save one representative video
- write the result into `TRAINING_STAGE_LOG.md`

### Rule 4. Action-validity metrics are mandatory

For multi-player Academy runs, always track:

- `invalid_ball_skill_rate`
- `invalid_no_ball_pass_rate`
- `invalid_no_ball_shot_rate`

If these stay high, the policy is still not learning valid football control.

## Current Operational Decision

The next PPO run should be:

- a fresh `academy_pass_and_shoot_with_keeper` reboot
- with `player_id`
- with the no-ball action filter enabled
- evaluated before any new `five_vs_five` attempt

## What Counts As Success

This reboot is successful if it produces a checkpoint that clearly shows:

- valid passes mostly happen from the ball holder
- shots are attempted after real buildup
- deterministic videos look like the task is being solved
- task completion improves, not just shaped return

Only then should that checkpoint be considered for later transfer back into `five_vs_five`.
