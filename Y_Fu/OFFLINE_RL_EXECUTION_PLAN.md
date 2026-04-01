# Offline RL Execution Plan

## Current Operating Rule

Do not start a new rollout-heavy GRF job while another major rollout-heavy job is already active.

Reason:

- PPO training is CPU-heavy
- offline data collection is also CPU-heavy
- running them together makes both slower and noisier

## Main Goal

Use offline RL as a `5_vs_5` improvement stage, not as a replacement for Academy curriculum.

The intended order is:

1. Academy PPO
2. `5_vs_5` PPO
3. `5_vs_5` offline RL
4. PPO fine-tuning from IQL if IQL looks better

## Main Rule About Data

Do not mix Academy and `5_vs_5` offline datasets in the same initial IQL run.

Reason:

- they are different tasks
- they have different controlled-player counts
- they have different state distributions
- naïve mixing is more likely to blur the objective than help it

Use Academy for PPO pretraining and `5_vs_5` for offline RL.

## Immediate Workflow

### Step 1: Finish Or Freeze The Current PPO Phase

If an Academy PPO phase is still active:

- finish it
- select the best checkpoint by metrics plus video

If a `5_vs_5` PPO phase is active:

- finish it
- choose one best checkpoint and one weaker checkpoint

If Academy is the next active phase:

- use the current Academy pass gates and hard-stop budgets
- do not extend Academy indefinitely just because the run is stable

Do not assume `latest.pt` is best.

### Step 2: Confirm Primitive Transfer In `5_vs_5`

Before offline RL, verify that the transferred `5_vs_5` PPO policy shows at least some of:

- simple passing
- support movement
- occasional shot creation

This check should happen after the early `5_vs_5` transfer window:

- roughly `250k ~ 500k env steps`
- roughly `1M ~ 2M agent steps`

If the `5_vs_5` PPO line still shows no meaningful primitive transfer, spend effort there first.

Offline RL is much more useful once the dataset already contains some real football behavior.

### Step 3: Run A Small `5_vs_5` Offline Pilot

Pilot target:

- `300K ~ 500K` env-steps total

Recommended sources:

1. best PPO checkpoint
   - `epsilon = 0.0`
2. best PPO checkpoint
   - `epsilon = 0.15`
3. weaker PPO checkpoint
   - `epsilon = 0.05`
4. random policy

Do not start full offline collection before this pilot is healthy.

Pilot purpose:

- verify manifest writing
- verify chunk loading
- verify timeout handling
- verify IQL training and evaluation
- verify shaped reward consistency

### Step 4: Run A Short IQL Smoke Test

Do not jump straight to `1M` gradient steps.

Start with:

- `20K` gradient steps
- evaluation every `5K`

Pilot success criteria:

1. dataset loads cleanly
2. IQL training runs end-to-end
3. best checkpoint is written
4. evaluation works
5. video and metrics are at least sane

### Step 5: Run Full `5_vs_5` IQL

Only after the pilot is healthy:

- collect the full `5_vs_5` offline corpus
- train `IQL iteration 0`
- evaluate the best IQL checkpoint

At this stage compare:

- current best `5_vs_5` PPO
- best `IQL iteration 0`

Evaluation should use:

- `20` deterministic episodes
- the same built-in opponent
- matching seeds when comparing runs

### Step 6: Decide Whether To Continue Offline

Continue the offline loop only if the IQL policy improves at least one of:

- visible passing quality
- shot creation
- goal difference
- early attack structure

If `IQL iteration 0` is not better than PPO, do not blindly collect another massive dataset.

### Step 7: If IQL Is Better, Use It In Two Ways

Option A:

- collect more `5_vs_5` data using the IQL checkpoint as the behavior policy
- retrain IQL on PPO plus IQL data

Option B:

- initialize PPO from the best IQL checkpoint
- resume online `5_vs_5` training

This is the desired replay-to-online loop.

## Selection Rules

### Best PPO checkpoint

Choose by:

1. `win_rate`
2. `avg_goal_diff`
3. visible passing and shot creation in video

### Weaker PPO checkpoint

Choose a checkpoint that is:

- clearly weaker than the best one
- still football-like
- not obviously broken

### Best IQL checkpoint

Choose by:

1. `win_rate`
2. `avg_goal_diff`
3. visible attack quality in video

## Acceptance Checklist

Do not move from pilot to full run unless all are true:

1. every dataset directory has a valid `manifest.json`
2. chunk files load without reconstruction errors
3. reward shaping in the manifest matches the source checkpoint family
4. IQL save and load works
5. `evaluate_iql.py` runs correctly
6. the pilot policy behavior is at least plausible

## Resource Rule

Before launching large collection:

- confirm enough disk remains for the dataset
- confirm no competing long PPO run is still active
- confirm the selected checkpoints are final enough to justify collection

## Practical Summary

The next operational question should not be:

- "Should I do Academy or offline RL?"

It should be:

- "Did Academy PPO create a good enough primitive?"
- "Did `5_vs_5` PPO transfer it?"
- "Can offline RL improve that `5_vs_5` policy further?"
- "Should the improved IQL policy go back into PPO?"

That is the intended execution logic for this repo.
