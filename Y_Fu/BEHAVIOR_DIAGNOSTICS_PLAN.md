# Behavior Diagnostics Plan

## Purpose

This note tracks the new idea that raw match metrics are not enough.

For the current `five_vs_five` line, we also need behavior-level diagnostics to detect degenerate policies early.

The main target is the failure mode:

- agents mostly drift or push to the right
- passing is too rare or too shallow
- movement is too single-directional
- the policy can "move" without looking like football

## Why This Matters

Videos already suggested that some failed runs:

- overuse forward/rightward movement
- fail to create enough meaningful passes
- fail to create enough shots

This is now treated as a measurable problem, not just an impression.

## Current Logged Diagnostics

The trainer now logs:

- `pass_rate`
- `shot_rate`
- `skill_rate`
- `idle_rate`
- `direction_rate`
- `right_bias`
- `left_bias`
- `vertical_bias`
- `top_actions`

These appear directly in the PPO training logs.

## How To Interpret Them

### `pass_rate`

Higher is not always better.

But if it stays too low, the policy is probably not coordinating enough.

### `shot_rate`

Still should remain relatively low overall, but if it is effectively zero for a long time, the policy is not finishing attacks.

### `direction_rate`

This measures how much of the action mass goes into directional movement actions.

If it dominates too strongly, the policy may be mostly steering rather than playing.

### `right_bias`

This is the most important new signal for the current failure.

It measures how much directional movement goes to the rightward family:

- `top_right`
- `right`
- `bottom_right`

If this stays very high for long periods, the policy is likely falling into "just move right" behavior.

### `left_bias`

This helps detect whether the policy can recycle, reset, or reposition instead of always forcing attack direction.

### `top_actions`

This is a compact view of the dominant actions in the rollout.

If the top actions are repeatedly:

- `right`
- `top_right`
- one pass action

then we are likely seeing directional over-bias.

## Practical Use

These diagnostics should be checked together with:

- `goals_for`
- `goals_against`
- `success_rate`
- representative videos

Do not use behavior diagnostics alone.

They are intended to explain why a run is failing, not replace task metrics.

## Immediate Rule

If a run shows all of these:

- `goals_for` pinned near zero
- repeated full-length matches
- high `right_bias`
- low effective pass and shot behavior

then the run should be treated as structurally degenerate, not just undertrained.

## Next Extension

If these metrics keep confirming the same failure, the next possible action is:

- add a light anti-degeneracy shaping term or movement-balance constraint

But that should happen only after the diagnostics have clearly confirmed the pattern.
