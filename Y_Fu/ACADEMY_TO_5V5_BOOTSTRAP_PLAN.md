# Academy-to-5v5 Bootstrap Plan

## Goal

Use Academy as a controlled PPO curriculum for primitive acquisition, then hand the policy off into `5_vs_5` before Academy overfitting becomes the main behavior.

The purpose of Academy is not to "solve football".

Its purpose is to make sure the policy already knows at least some of:

- ball-to-goal progression
- simple pass release
- receiving and finishing
- one extra layer of support behavior

The final judgment still happens in `5_vs_5`.

## Core Claim

The current evidence suggests that pure `5_vs_5` PPO from scratch is too hard in this repo for primitive acquisition.

The practical interpretation is:

- `5_vs_5` is asking PPO to learn too many things at once
- Academy can reduce the exploration and credit-assignment burden
- Academy should therefore be treated as a bootstrap stage
- Academy should not absorb most of the total compute budget

So the intended line is:

1. learn the primitive in Academy
2. pick the best Academy checkpoint by behavior, not by recency
3. transfer it into `5_vs_5`
4. spend the main training budget in `5_vs_5`

## One Important Accounting Rule

This repo's PPO trainer uses `total_timesteps` as `agent steps`, not raw environment transitions.

For this plan, use both units:

- `env steps`: one simulator transition
- `agent steps`: `env_steps * num_controlled_players`

Approximate conversion:

- `academy_run_to_score_with_keeper`: `1 env step = 1 agent step`
- `academy_pass_and_shoot_with_keeper`: `1 env step = 2 agent steps`
- `academy_3_vs_1_with_keeper`: `1 env step = 3 agent steps`
- `5_vs_5`: `1 env step = 4 agent steps`

This matters because curriculum budgets should be compared mostly in `env steps`, not only in raw `total_timesteps`.

## What Academy Should Teach

### Stage 1: `academy_run_to_score_with_keeper`

This stage should teach:

- direct attack intent
- decisive ball carry toward goal
- finishing against the keeper

This is not mainly a passing stage.

It is a quick way to make sure the encoder and PPO loop learn basic attack geometry.

### Stage 2: `academy_pass_and_shoot_with_keeper`

This stage should teach:

- passing as a useful action, not just dribbling
- support positioning for the receiver
- pass-then-finish structure

This is the most important Academy stage for the current branch.

### Stage 3: `academy_3_vs_1_with_keeper`

This stage should teach:

- using numerical advantage
- not forcing the first lane blindly
- one more layer of simple support and decision making

This is the last Academy step before `5_vs_5`.

## Budget Recommendation In `env steps`

The current default presets are too short to be treated as a serious curriculum budget.

Use the presets as starting points, but expect to extend them.

### Recommended budget table

| Stage | Controlled Players | Pilot Budget | Normal Budget | Hard Stop |
|---|---:|---:|---:|---:|
| `academy_run_to_score_with_keeper` | 1 | `150k env steps` | `250k ~ 400k env steps` | `600k env steps` |
| `academy_pass_and_shoot_with_keeper` | 2 | `150k env steps` | `300k ~ 600k env steps` | `900k env steps` |
| `academy_3_vs_1_with_keeper` | 3 | `150k env steps` | `300k ~ 700k env steps` | `1.0M env steps` |

Equivalent `total_timesteps` to pass into PPO:

- Stage 1: `150k ~ 600k`
- Stage 2: `300k ~ 1.8M`
- Stage 3: `450k ~ 3.0M`

Interpretation:

- pilot budget: enough to see whether the stage is alive
- normal budget: the range where the stage should usually pass if the reward is right
- hard stop: if it still looks bad here, do not just keep training blindly

## Budget-To-Command Conversion

Use this conversion when launching runs:

- Stage 1 target `250k env steps` -> `--total-timesteps 250000`
- Stage 2 target `400k env steps` -> `--total-timesteps 800000`
- Stage 3 target `500k env steps` -> `--total-timesteps 1500000`

Concrete examples:

```bash
python -u Y_Fu/train.py --preset academy_run_to_score_with_keeper --total-timesteps 250000 --device cpu
python -u Y_Fu/train.py --preset academy_pass_and_shoot_with_keeper --total-timesteps 800000 --device cpu
python -u Y_Fu/train.py --preset academy_3_vs_1_with_keeper --total-timesteps 1500000 --device cpu
```

If Stage 2 or Stage 3 is continuing from an earlier checkpoint, keep the same stage preset and set:

- `--init-checkpoint <best_checkpoint>`

The main rule is:

- pick the budget in `env steps`
- convert it to `total_timesteps` using the controlled-player count
- evaluate multiple checkpoints inside the run instead of trusting the final one automatically

## Training Shape

Use Academy as short, strict, and checkpoint-driven PPO training.

Recommended operational shape:

1. start from the current preset
2. raise `total_timesteps` to match the target `env step` budget
3. evaluate every named checkpoint block
4. stop early once the stage clearly passes
5. if the stage is still weak near the hard stop, change reward or handoff logic rather than only adding more steps

For stability, keep the basic PPO shape close to the current implementation:

- `representation=extracted`
- `model_type=cnn`
- `feature_dim=256`
- `update_epochs=4` for multi-player Academy
- `num_minibatches=4`
- `learning_rate=3e-4`
- moderate rollout sizes: `128 / 192 / 256`

Recommended practical detail:

- use `num_envs=4 ~ 6` for Academy
- keep `save_interval` frequent enough that you can inspect several checkpoints inside one run

## Reward Design For Academy

The reward in Academy should be shaped toward short attacking sequences, not generic possession.

That means the main reward logic is:

- passing should be encouraged
- pass progress should matter
- shots should be encouraged
- turnovers should hurt
- passive possession should stay small

### Stage 1 reward rule

For `academy_run_to_score_with_keeper`, keep reward simple.

Default recommendation:

- use built-in `scoring,checkpoints`
- do not add heavy dense shaping unless the stage is clearly stuck

Reason:

- this stage is mostly about direct carry-and-finish behavior
- too much extra shaping here is unnecessary and may bias behavior in a way that does not help later stages

### Stage 2 reward rule

For `academy_pass_and_shoot_with_keeper`, the reward should explicitly teach:

- pass completion
- forward pass usefulness
- shot creation

The current preset is a reasonable starting point:

- `pass_success_reward = 0.08`
- `pass_failure_penalty = 0.04`
- `pass_progress_reward_scale = 0.08`
- `shot_attempt_reward = 0.03`
- `final_third_entry_reward = 0.04`
- `own_half_turnover_penalty = 0.02`

Background shaping should stay small:

- `attacking_possession_reward = 0.002`
- `possession_retention_reward = 0.001`

Practical rule:

- if the policy keeps the ball but still does not score, reduce passive possession terms before increasing pass reward further
- if the policy shoots too little, raise `shot_attempt_reward` slightly before making pass reward much larger

### Stage 3 reward rule

For `academy_3_vs_1_with_keeper`, keep the same reward family, but the interpretation changes:

- pass reward should still exist
- shot creation still matters
- blind possession should not dominate
- the extra attacker should be used as a real second option

The current preset is again a reasonable starting point:

- `pass_success_reward = 0.08`
- `pass_failure_penalty = 0.05`
- `pass_progress_reward_scale = 0.08`
- `shot_attempt_reward = 0.03`
- `final_third_entry_reward = 0.04`
- `own_half_turnover_penalty = 0.02`

Practical rule:

- if Stage 3 looks like endless safe circulation or weak dribbling, reduce possession-style bonuses before adding more compute

## Completion Standards

Training completion is not task completion.

Each Academy stage needs both:

- metric threshold
- qualitative behavior check

### Stage 1 pass standard

Use:

- `20` deterministic episodes across multiple seeds

Pass standard:

- scored-episode ratio `>= 0.80`
- `avg_score_reward >= 0.80`

Video standard:

- the agent moves directly to goal with little hesitation
- obvious finishing chances are usually converted

### Stage 2 pass standard

Use:

- `20` deterministic episodes across multiple seeds, or
- `50` episodes in a single-seed sweep for checkpoint comparison

Pass standard:

- scored-episode ratio `>= 0.60`
- `avg_score_reward >= 0.60`
- `win_rate >= 0.60`

Video standard:

- pass-to-shot structure appears repeatedly
- the ball carrier does not force dribble-only behavior on obvious 2-player attacks
- the receiver is visibly part of the attack, not just a bystander

### Stage 3 pass standard

Use:

- `20` deterministic episodes across multiple seeds

Pass standard:

- scored-episode ratio `>= 0.55`
- `avg_score_reward >= 0.55`

Video standard:

- the policy uses the extra attacker in a meaningful fraction of attacks
- the first blocked lane does not always kill the whole play
- the attack ends in shots often enough that it looks purposeful rather than random

### Failure standard

Treat an Academy stage as failed for the current configuration if one of these is true:

- it reaches the hard-stop budget and still does not clear the pass gate
- return increases but scoring behavior still looks weak
- later checkpoints look worse than earlier ones and nothing transfers cleanly

That is the point to adjust:

- reward shaping
- checkpoint selection
- or the transfer plan

not just the budget.

## Handoff Rule Into `5_vs_5`

The transfer checkpoint should be:

1. best passed Stage 3 checkpoint if Stage 3 really passed
2. otherwise best passed Stage 2 checkpoint
3. otherwise best Stage 2 checkpoint with the cleanest visible pass-to-shot behavior
4. otherwise best Stage 1 checkpoint only as a weak fallback

Important:

- do not transfer `latest.pt` automatically
- do not transfer the highest-return checkpoint automatically
- choose the checkpoint that looks most likely to survive contact with `5_vs_5`

## How Academy Reward Should Extend Into `5_vs_5`

Do not copy the Academy objective into `5_vs_5` unchanged.

The role of Academy reward is:

- teach the primitive

The role of `5_vs_5` reward is:

- preserve the primitive while making it useful in a longer and noisier game

So the transition rule should be:

- keep pass and shot encouragement in `5_vs_5`
- reduce the relative weight of easy possession-style bonuses
- make sure `5_vs_5` shaping does not reward empty ball movement more than actual attack construction

This is already consistent with the current `five_vs_five` preset, where:

- pass reward is smaller than in Academy
- shot and final-third incentives are still present
- passive possession terms are reduced

That is the correct direction.

## `5_vs_5` Transfer Plan

The first question after Academy is not "did we win `5_vs_5` already?"

The first question is:

- did the Academy primitive survive transfer?

### Transfer phase A: early check

Start `5_vs_5` from the best Academy checkpoint and inspect the first:

- `250k ~ 500k env steps`
- equivalent to `1M ~ 2M agent steps` in `5_vs_5`

What to check:

- does passing still appear at all
- does the team create more shots than scratch PPO
- do attacks collapse immediately into dribble chaos
- are matches still all full-length low-quality losses

### Transfer phase B: direct baseline comparison

Run a small comparison:

1. `5_vs_5` from best Academy checkpoint
2. `5_vs_5` from scratch

Compare them at matched early budgets:

- `250k env steps`
- `500k env steps`

Preferred signals:

- `pass_rate`
- `shot_rate`
- video evidence of support behavior
- fewer degenerate movement patterns

If the transferred run is clearly cleaner early, Academy is earning its place.

### Transfer phase C: main `5_vs_5` budget

Only after the transfer looks real should the policy receive the long `5_vs_5` budget:

- `2.5M ~ 5M env steps`
- equivalent to `10M ~ 20M agent steps`

That keeps the main compute where it belongs.

## What Counts As Academy Success

Academy is worth keeping only if it improves at least one real `5_vs_5` outcome:

- earlier appearance of passing
- higher shot creation in early `5_vs_5`
- cleaner attack structure in video
- better sample efficiency than scratch PPO

Academy is not justified if:

- it only produces nice scripted videos inside Academy
- transfer to `5_vs_5` is weak
- the same `5_vs_5` behavior can be reached just as fast from scratch

## Recommended Current Default

For the current repo state, the most reasonable Academy line is:

1. `academy_run_to_score_with_keeper` for a short budget only
2. make `academy_pass_and_shoot_with_keeper` the main Academy stage
3. use `academy_3_vs_1_with_keeper` as the last transfer filter, not as an endless compute sink
4. transfer into `5_vs_5` as soon as Stage 2 or Stage 3 shows real repeatable pass-to-shot behavior

## Bottom Line

The compact rule is:

- Academy should teach the primitive
- Academy should have a real but capped `env step` budget
- Academy completion must be judged by both metrics and video
- the checkpoint handoff must be deliberate
- real success is whether the primitive survives and helps in `5_vs_5`
