# Five-v-Five Execution Checklist

## Purpose

This is the concrete execution checklist for fixing the current `five_vs_five` failure mode.

It is not a brainstorming note.

It is the operational sequence to follow from here.

## Current Situation

The current line has already shown a strong failure signal:

- the old `five_vs_five` run crossed `10M+` agent steps
- it still produced behavior that did not look like football
- the policy often just moved, chased, or pushed forward without meaningful organization

Because of that, the next phase must be:

- controlled diagnosis
- minimal-change experiments
- strict stop criteria

## Resource Rules

These rules are here to avoid wasting CPU and memory.

### Main training rule

- run only one main training job at a time

### Environment count rule

- default to `num_envs=4` for `five_vs_five`
- do not raise back to `6` or `8` unless memory remains clearly stable for a long window

### Evaluation rule

- only run short evaluations between training checkpoints
- do not run another long training job in parallel with the active one

### Video rule

- save representative videos only at named checkpoints
- do not dump videos continuously during training

### Storage rule

- keep local checkpoints local
- do not commit personal checkpoint directories
- only commit markdown, README updates, and clearly curated outputs

## Active Run

Current active run:

- experiment: `five_vs_five_reward_v1_e4`
- purpose: first reward-only revision
- config:
  - `preset=five_vs_five`
  - `num_envs=4`
  - `rollout_steps=256`
  - `total_timesteps=10_000_000`
  - `update_epochs=4`
  - `num_minibatches=1`
  - warm start: `academy_pass_and_shoot_with_keeper/update_10.pt`

Current interpretation:

- this run is the first test of whether reward-only revision changes the failure pattern
- it should be allowed to reach at least the first meaningful diagnosis checkpoint
- it should not be allowed to consume unlimited budget if the behavior remains structurally wrong

## Execution Order

Follow this order unless a run crashes or becomes obviously invalid.

### Step 1. Finish the reward-only revision diagnosis

Run:

- current experiment: `five_vs_five_reward_v1_e4`

Primary question:

- does reward-only revision make the policy less empty, less passive, and more goal-oriented?

Mandatory checkpoints:

1. around `1M` agent steps
2. around `2M` agent steps
3. around `5M` agent steps

Record at each checkpoint:

- checkpoint path
- wall-clock time
- total agent steps
- `goals_for`
- `goals_against`
- `avg_goal_diff`
- `win_rate`
- one representative video if the checkpoint is important

Immediate stop condition:

- if by `2M` the run still shows:
  - `goals_for` effectively pinned at zero
  - `success_rate` pinned near zero
  - repeated `3001`-step full-length losses
  - no visible increase in shots or dangerous attacks

then:

- stop treating reward-only revision as sufficient
- move to Step 2

Continue condition:

- if by `2M` or `5M` the run shows:
  - more real attacking sequences
  - more nonzero goal events
  - better scorelines
  - less empty full-length drifting

then:

- let it continue to the next checkpoint

### Step 2. Add `player_id`

This is the next highest-priority intervention if reward-only revision is not enough.

Goal:

- reduce role collapse under the shared policy

Implementation target:

- keep parameter sharing
- append explicit player identity input to each controlled player observation

Minimal version:

- use one-hot `player_id`

Why this comes second:

- the current failure strongly looks like shared-policy homogeneity
- this is still much cheaper than algorithm replacement

Stop condition:

- if reward-only revision already creates clearly better football behavior, do not rush this change
- otherwise, implement it next

### Step 3. Reward ablation table

Do this after either:

- reward-only revision fails
- or reward-only revision shows partial improvement but not enough

Keep the table small.

Recommended first 3 variants:

1. current reward revision baseline
2. same as baseline, but reduce pass rewards further
3. same as baseline, but increase shot reward and final-third reward slightly

Do not explode into too many runs.

Goal:

- identify which reward family is actually helping

### Step 4. Warm-start selection

After at least one revised reward run and possibly one `player_id` run:

- compare early transfer from multiple warm starts
- select the checkpoint that gives the best early `five_vs_five` behavior

Do not pick warm starts only by curriculum return.

### Step 5. Small PPO sweep

Only after Steps 1 to 4.

Try a small sweep over:

- entropy coefficient
- rollout length
- learning rate

Keep the sweep tiny.

Do not start broad hyperparameter search before fixing objective and role structure.

### Step 6. Opponent pool / light self-play

Only after the policy starts to play recognizably better football against the built-in bot.

Do not use this as the immediate answer to the current failure.

## Detailed Stop Gates

These gates are here to protect compute.

### Gate A. Reward-only revision gate

At `2M` steps:

- `No-go` if:
  - `goals_for` is still near zero
  - `success_rate` still near zero
  - videos still look like empty forward drift or ball collapse

- `Borderline` if:
  - there are more attacks but still few goals
  - behavior looks less chaotic than the old 10M failure

- `Go` if:
  - goal events start appearing
  - scorelines improve
  - video quality improves visibly

### Gate B. `player_id` gate

After implementing `player_id`, check whether:

- off-ball support improves
- spacing improves
- multiple players stop collapsing onto the same point

If not, the next issue is likely not only identity but also objective and optimization.

### Gate C. Reward-table gate

If no reward family clearly beats the others on:

- `win_rate`
- `avg_goal_diff`
- video quality

then the project should stop pretending reward tuning alone will solve it.

That is the point to consider stronger structural changes.

## Minimal CPU-Aware Workflow

Use this default workflow:

1. keep one active training run
2. poll it periodically
3. only run short deterministic evaluation when a named checkpoint is reached
4. save one representative video for key checkpoints only
5. write the outcome into `TRAINING_STAGE_LOG.md`

Avoid:

- two long training runs in parallel
- continuous evaluation during training
- video dumping for every checkpoint
- large parameter sweeps before the first structural fixes

## Required Documentation Updates

For each important checkpoint:

1. update `TRAINING_STAGE_LOG.md`
2. save or link one representative video if it is a major checkpoint
3. if the result changes project direction, summarize it in a dedicated markdown
4. keep `README.md` document map current

Failures must be logged explicitly.

Do not log only successes.

## Immediate Next Actions

From this exact point, the next actions are:

1. let the current reward-only revision run continue to the next diagnosis checkpoint
2. evaluate whether it differs meaningfully from the old `10M` failure pattern
3. if not, implement `player_id`
4. after that, run a small reward ablation table

## Final Priority Order

The concrete priority order is:

1. reward-only revision diagnosis
2. `player_id`
3. reward ablation table
4. better warm-start selection
5. small PPO hyperparameter sweep
6. light self-play or opponent pool

This order is the current default execution policy for the `Y_Fu` `five_vs_five` line.
