# Five-v-Five Half-Day Checkpoint

## Purpose

Use this checkpoint after roughly `6 ~ 9` hours of `five_vs_five` training to decide whether the current run is worth continuing.

This is a `go / no-go` check, not a paper-grade evaluation.

## Current Training Target

- preset: `five_vs_five`
- envs: `6`
- rollout_steps: `256`
- total_timesteps: `20_000_000`
- update_epochs: `4`
- num_minibatches: `1`
- init checkpoint: `Y_Fu/checkpoints/academy_pass_and_shoot_with_keeper/update_10.pt`

## When To Check

Check at these points:

1. Early trend check: around `5M` agent steps
2. Main decision point: around `10M` agent steps
3. End of current run: around `20M` agent steps

If the run looks clearly bad at `10M`, do not blindly spend the rest of the budget.

## What To Record

For each check:

- checkpoint path
- wall-clock time
- total agent steps
- `win_rate`
- `avg_goal_diff`
- `avg_score_reward`
- one representative video

## Pass / Fail Heuristics

### At 5M Steps

Good signs:

- no longer dominated by flat `0-0` episodes
- some real goal events appear
- videos show stable ball progression instead of random wandering
- occasional recovery and reset in defense

Warning signs:

- still mostly `0-0`
- attack stalls near midfield
- multiple players collapse onto the ball
- no clear final-third entries

### At 10M Steps

`Go` if most of these are true:

- `win_rate >= 0.25`
- `avg_goal_diff >= -0.30`
- videos show repeated structured attacks
- some shots or scoring sequences are reproducible

`Borderline` if:

- `win_rate` is around `0.15 ~ 0.25`
- `avg_goal_diff` is still negative but improving
- play quality looks visibly better than early training

`No-go` if most of these are true:

- `win_rate < 0.15`
- `avg_goal_diff < -0.50`
- video still looks mostly random or passive
- matches remain dominated by `0-0` or harmless possession

### At 20M Steps

Worth continuing this configuration if:

- `win_rate >= 0.35`
- `avg_goal_diff >= -0.10`
- behavior looks like real `5v5` play rather than shaped-reward gaming

Stop and change setup if:

- `win_rate` is still very low
- goal difference is clearly negative
- progress from `10M` to `20M` is small

## If The Run Fails

Change one of these before the next long run:

1. reward shaping
2. warm-start checkpoint choice
3. explicit player identity input

Do not respond to a failed half-day check by only increasing timesteps.
