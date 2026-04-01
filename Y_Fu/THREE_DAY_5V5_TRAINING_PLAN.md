# 3-Day 5v5 Training Plan

## Goal

The goal is to train a stronger `5_vs_5` multi-agent player in Google Research Football using the current shared-policy PPO pipeline in `Y_Fu/yfu_football`.

This plan is intentionally centered on:

- parallel environment sampling
- stable PPO updates
- Academy warm-up only as preparation
- spending most of the compute budget on `5_vs_5`

It is not centered on:

- `11_vs_11` as the main target
- immediate self-play
- large algorithm refactors

## Current Environment Setup

### What we are training right now

In the current `five_vs_five` preset:

- scenario: `5_vs_5`
- controlled players: 4 left-side outfield players
- goalkeeper: controlled by the environment, not by the policy
- opponent: GRF built-in scripted bot
- observation: `extracted`
- rewards: `scoring,checkpoints` plus local custom reward shaping

This means the current system is:

- not self-play
- not full-team control including the keeper
- still a valid multi-agent setup with a shared policy over multiple controlled players

### Why `num_controlled_players=4` is correct in `5_vs_5`

The GRF `5_vs_5` scenario defines each goalkeeper as `controllable=False`, so the 5 players are:

- 1 built-in goalkeeper
- 4 controllable outfield players

So for this scenario, controlling 4 players is the intended setup.

## Training Philosophy

### Run completion is not stage success

From this point on, every stage should be tracked with two separate labels:

- `run completed`: the configured training budget finished
- `stage passed`: evaluation cleared the stage threshold

Only `stage passed` should unlock the next stage in a strict progression.

If a stage finishes its budget but does not pass:

- extend the budget, or
- adjust shaping / hyperparameters, or
- keep the checkpoint only as exploratory context

### Why we are not starting with self-play

At this stage, the built-in GRF opponent is still useful because:

- it provides a stable baseline
- it makes debugging and PPO tuning easier
- it avoids introducing opponent-learning instability too early

We should move toward self-play or an opponent pool later, but only after the `5_vs_5` policy can reliably beat the built-in opponent.

### When self-play becomes worth adding

Start considering self-play or an opponent pool when `5_vs_5` training shows:

- stable positive goal difference
- repeated positive performance across seeds
- sustained high win rate against the built-in bot

A better transition target than pure self-play is:

- built-in bot
- historical checkpoints
- strongest recent checkpoint snapshots

## Budget Profiles

The schedule below is a stable workflow template. If the goal is to better match actual wall-clock budget, use one of these profiles.

### Light Profile

Use this when:

- verifying the full pipeline
- testing a new reward setting
- checking whether `num_envs` and rollout settings are stable

Recommended total scale:

- academy warm-up: `1M ~ 2M agent steps`
- `five_vs_five`: `5M ~ 10M agent steps`

### One-Day Profile

Use this when:

- you want a serious `5_vs_5` run
- but still want room for restarts, evaluation, or a second seed

Recommended total scale:

- academy warm-up: `1M ~ 3M agent steps`
- `five_vs_five`: `10M ~ 20M agent steps`

### Two-To-Three-Day Profile

Use this when:

- the machine is mostly dedicated to this task
- the goal is to spend most of the budget on the actual `5_vs_5` target

Recommended total scale:

- academy warm-up: keep total at `1M ~ 3M agent steps`
- `five_vs_five` main training: increase to `20M ~ 50M agent steps`

With the observed throughput range of roughly `1500 ~ 1800 samples/sec`:

- `20M steps` is on the order of `3 ~ 5 hours`
- `50M steps` is on the order of `8 ~ 12 hours`

In practice, the full 2 to 3 day budget is naturally consumed by:

- multiple seeds
- reruns after small hyperparameter changes
- pauses for evaluation
- switching from built-in bot training to stronger opponent training later

### Recommended Current Choice

For the current project stage, use:

- Academy warm-up as already planned
- `five_vs_five` expanded to the Two-To-Three-Day profile

That means the most useful expansion is not more Academy time, but a larger `five_vs_five` block.

## PPO Settings To Favor Stability

Based on the current PPO implementation and the papers in `refs`, use:

- `update_epochs=4`
- `num_minibatches=1`
- moderate clip ratio, keep current `clip_coef=0.2`

Reason:

- multi-agent PPO often degrades when samples are reused too much
- fewer minibatches and fewer epochs are generally more stable
- larger on-policy rollout batches should come from parallel environments, not from more aggressive sample reuse

## Stage Passing Gates

These are the operational gates for deciding whether a stage is actually good enough to hand off.

### Stage 1: `academy_run_to_score_with_keeper`

Evaluation recommendation:

- `20` deterministic episodes across multiple seeds

Pass gate:

- scored-episode ratio `>= 0.80`

### Stage 2: `academy_pass_and_shoot_with_keeper`

Evaluation recommendation:

- `20` deterministic episodes across multiple seeds, or
- `50` total episodes in a single-seed sweep

Pass gate:

- scored-episode ratio `>= 0.60`

Important note:

- shaped return alone is not enough
- if the video still shows repeated non-scoring possession, the stage is not passed

### Stage 3: `academy_3_vs_1_with_keeper`

Evaluation recommendation:

- `20` deterministic episodes across multiple seeds

Pass gate:

- scored-episode ratio `>= 0.55`

### Stage 4: `five_vs_five`

Evaluation recommendation:

- `20` deterministic episodes across multiple seeds against the built-in bot

Pass gate:

- `win_rate >= 0.35`
- `avg_goal_diff >= -0.10`

### Stage 5: `five_vs_five`

Evaluation recommendation:

- `20` deterministic episodes across multiple seeds against the built-in bot

Pass gate:

- `win_rate >= 0.55`
- `avg_goal_diff > 0.00`

## 3-Day Schedule

The machine budget discussed during planning:

- about 2 to 3 days
- about 200G storage
- about 6G CPU memory
- at most 10 WSL workers

Practical recommendation:

- use `num_envs=6` as the default stable setting
- test `num_envs=8` only on lighter Academy tasks or after confirming headroom
- do not jump to `num_envs=10` immediately

### Stage 0: Smoke Test

Purpose:

- verify vectorized rollout
- verify GRF engine/import path
- measure rough throughput

Configuration:

- preset: `academy_pass_and_shoot_with_keeper`
- `num_envs=6`
- `rollout_steps=128`
- `total_timesteps=80_000`
- `save_interval=2`
- `update_epochs=4`
- `num_minibatches=1`

Status:

- completed
- latest checkpoint: `Y_Fu/checkpoints/academy_pass_and_shoot_with_keeper/latest.pt`
- this should still be treated as a smoke-test stage, not a pass/fail gate for the full curriculum

Observed behavior:

- rollout throughput was roughly `env_fps=700~1000`
- `samples_per_sec` was roughly `1400~2000`
- the policy began to produce occasional goals, but performance is still early-stage

### Stage 1: Simple Progression Warm-up

Purpose:

- reinforce direct ball carrying and scoring instincts
- stabilize early value learning on an easier task

Configuration:

- preset: `academy_run_to_score_with_keeper`
- `num_envs=8`
- `rollout_steps=128`
- `total_timesteps=300_000`
- `save_interval=5`
- `update_epochs=4`
- `num_minibatches=1`

Run command:

```bash
.venv_yfu_grf_sys/bin/python Y_Fu/train.py \
  --preset academy_run_to_score_with_keeper \
  --num-envs 8 \
  --rollout-steps 128 \
  --total-timesteps 300000 \
  --save-interval 5 \
  --update-epochs 4 \
  --num-minibatches 1 \
  --seed 42
```

### Stage 2: Passing and Finish Warm-up

Purpose:

- train short passing, second-run timing, and final shot behavior
- prepare for the 4-player outfield coordination pattern used in `5_vs_5`

Configuration:

- preset: `academy_pass_and_shoot_with_keeper`
- `num_envs=8`
- `rollout_steps=192`
- `total_timesteps=800_000`
- `save_interval=5`
- `update_epochs=4`
- `num_minibatches=1`
- `init_checkpoint=Y_Fu/checkpoints/academy_pass_and_shoot_with_keeper/latest.pt`

Run command:

```bash
.venv_yfu_grf_sys/bin/python Y_Fu/train.py \
  --preset academy_pass_and_shoot_with_keeper \
  --num-envs 8 \
  --rollout-steps 192 \
  --total-timesteps 800000 \
  --save-interval 5 \
  --update-epochs 4 \
  --num-minibatches 1 \
  --init-checkpoint Y_Fu/checkpoints/academy_pass_and_shoot_with_keeper/latest.pt \
  --seed 42
```

### Stage 3: Small-Team Coordination Warm-up

Purpose:

- increase off-ball movement and three-player coordination
- improve behavior before entering `5_vs_5`

Configuration:

- preset: `academy_3_vs_1_with_keeper`
- `num_envs=6`
- `rollout_steps=192`
- `total_timesteps=1_200_000`
- `save_interval=5`
- `update_epochs=4`
- `num_minibatches=1`

Run command:

```bash
.venv_yfu_grf_sys/bin/python Y_Fu/train.py \
  --preset academy_3_vs_1_with_keeper \
  --num-envs 6 \
  --rollout-steps 192 \
  --total-timesteps 1200000 \
  --save-interval 5 \
  --update-epochs 4 \
  --num-minibatches 1 \
  --seed 42
```

### Stage 4: Main 5v5 Training Block 1

Purpose:

- shift most of the compute budget onto the real target
- start learning robust `5_vs_5` play instead of isolated Academy skills

Configuration:

- preset: `five_vs_five`
- `num_envs=6`
- `rollout_steps=256`
- `total_timesteps=10_000_000`
- `save_interval=10`
- `update_epochs=4`
- `num_minibatches=1`
- initialize from the best warm-up checkpoint

Recommended initial checkpoint order:

1. best `academy_3_vs_1_with_keeper`
2. otherwise best `academy_pass_and_shoot_with_keeper`
3. otherwise best `academy_run_to_score_with_keeper`

Important rule:

- choose `best` by evaluation, not by latest timestamp
- do not promote a checkpoint into `five_vs_five` just because the training run finished

Run command template:

```bash
.venv_yfu_grf_sys/bin/python Y_Fu/train.py \
  --preset five_vs_five \
  --num-envs 6 \
  --rollout-steps 256 \
  --total-timesteps 10000000 \
  --save-interval 10 \
  --update-epochs 4 \
  --num-minibatches 1 \
  --init-checkpoint <best_warmup_checkpoint> \
  --seed 42
```

### Stage 5: Main 5v5 Training Block 2

Purpose:

- continue `5_vs_5` optimization once the policy is no longer random
- optionally increase collection throughput if the machine remains stable

Configuration:

- preset: `five_vs_five`
- `num_envs=6`, then try `8` only if memory and stability remain acceptable
- `rollout_steps=256` or `384`
- `total_timesteps=10_000_000 ~ 40_000_000`
- `save_interval=10`
- `update_epochs=4`
- `num_minibatches=1`

Recommended rule:

- if `num_envs=6` stays stable and CPU is still underused, test `num_envs=8`
- if `num_envs=8` slows down sharply or becomes unstable, fall back to `6`

## Practical Decision Rules

### If CPU looks idle

Increase in this order:

1. `num_envs`
2. `rollout_steps`

Do not increase in this order:

1. `num_minibatches`
2. `update_epochs`

### If RAM becomes tight or the environment gets unstable

Reduce in this order:

1. `num_envs`
2. `rollout_steps`

### If 5v5 learning is too attack-heavy

Current warm-up tasks are offense-biased. If needed later:

- spend more time in `5_vs_5`
- consider adding `academy_counterattack_hard`
- consider slightly stronger defensive shaping

## Mid-Term Improvement Ideas

### Player identity input

One likely next step is adding `player_id` conditioning to the shared policy.

Current policy:

- shared network
- different observations
- no explicit player identity

Potential improved policy:

- shared network
- different observations
- explicit `player_id` one-hot or embedding

Why this may help:

- reduce homogeneous behavior
- improve role differentiation
- improve off-ball movement and coverage responsibilities

This is not urgent for the current 3-day run, but it is a strong candidate for the next iteration.

### Self-play later

Do not make self-play the main training source immediately.

Better future progression:

1. built-in bot baseline
2. mixed opponents
3. opponent pool / self-play

Suggested later opponent mix:

- 50% built-in bot
- 30% historical checkpoints
- 20% strongest recent snapshots

## Commands and Environment Notes

Use the local environment that successfully ran the smoke test:

```bash
.venv_yfu_grf_sys/bin/python
```

The path fix for `gfootball_engine` is already applied in:

- `Y_Fu/yfu_football/envs.py`

So the current training flow should work from this repository without extra manual path edits.

## Recommended Immediate Next Step

The immediate operational rule is:

- keep training and evaluating the current warm-up stages
- do not mark a stage as solved unless it clears its pass gate
- only enter `five_vs_five` as the main handoff once a warm-up checkpoint actually passes, or once we explicitly choose to proceed despite a failed gate
