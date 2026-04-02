# Five-v-Five Reward V2 Ablation Table

## Purpose

This file turns the reward-v2 idea into a small controlled ablation.

The goal is not to search a huge coefficient space.

The goal is to answer one practical question:

- when we stop rewarding empty play, what balance between attack completion, progression quality, and transition discipline produces the best early `5_vs_5` behavior?

The three presets are all implemented in [ppo.py](/home/yuhan/Codes/RL/Learning-to-Play-Football-Multi-Agent-Reinforcement-Learning/Y_Fu/yfu_football/ppo.py).

## The Three Presets

### 1. `five_vs_five_reward_v2`

This is the new default attack-quality version.

Core idea:

- keep pass reward small
- make final-third entry and shot creation the main dense attacking terms
- keep only the clearest turnover penalty

Best use:

- default first candidate

### 2. `five_vs_five_reward_v2b_transition`

This is the more conservative transition-discipline version.

Core idea:

- weaken pass reward slightly
- increase `own_half_turnover_penalty`
- increase `defensive_third_recovery_reward`

Best use:

- if the base v2 line looks too reckless
- if videos show repeated self-inflicted dangerous turnovers
- if `own_half_turnovers_ep` and `opponent_dangerous_possessions_ep` stay too high

### 3. `five_vs_five_reward_v2c_progression`

This is the progression-quality test.

Core idea:

- keep generic possession at zero
- slightly strengthen `pass_progress_reward_scale`
- keep shot and final-third reward strong, but not maximal

Best use:

- if the base v2 line still cannot move the ball into danger often enough
- if `final_third_entries_ep` stays too low

## Coefficient Table

| Preset | pass_success | pass_failure | pass_progress | shot_attempt | final_third | possession_retention | own_half_turnover | defensive_third_recovery |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `five_vs_five_reward_v2` | `0.02` | `0.02` | `0.02` | `0.08` | `0.08` | `0.00` | `0.02` | `0.015` |
| `five_vs_five_reward_v2b_transition` | `0.015` | `0.02` | `0.015` | `0.08` | `0.08` | `0.00` | `0.03` | `0.020` |
| `five_vs_five_reward_v2c_progression` | `0.02` | `0.02` | `0.035` | `0.07` | `0.07` | `0.00` | `0.02` | `0.015` |

Shared across all three:

- `attacking_possession_reward = 0.0`
- `possession_retention_reward = 0.0`
- `opponent_attacking_possession_penalty = 0.0`
- `possession_recovery_reward = 0.01`
- `rewards = "scoring,checkpoints"`

## What Each Variant Is Testing

### `v2`

Question:

- if we strongly prioritize danger creation and shot conversion, does the policy finally stop playing empty football?

Main expected signals:

- higher `final_third_entries_ep`
- higher `shot_attempt_events_ep`
- better `shot_per_final_third_entry`

### `v2b_transition`

Question:

- is the current line failing partly because it attacks too carelessly and gets punished in transition?

Main expected signals:

- lower `own_half_turnovers_ep`
- lower `opponent_dangerous_possessions_ep`
- maybe slower attack growth, but better stability

### `v2c_progression`

Question:

- does the line still need more credit for meaningful forward movement, even after generic possession reward is removed?

Main expected signals:

- higher `final_third_entries_ep`
- possibly higher `pass_progress_ep`
- attack gets into danger more often, without collapsing into pass-cycling

## Early-Stop Evaluation Rule

Do not run these overnight immediately.

Use the same early transfer check for all three:

- start from the same Academy checkpoint
- run to roughly `2M` agent steps
- compare the same metrics

Kill a variant early if it still shows:

- `goals_for` pinned at zero
- mostly full-length `3001` matches
- low `final_third_entries_ep`
- low `shot_attempt_events_ep`
- high `right_bias`
- degenerate `top_actions`

## Suggested Command Pattern

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

## Ranking Rule

Do not rank by shaped return first.

Rank variants in this order:

1. `goals_for`
2. `avg_goal_diff`
3. `final_third_entries_ep`
4. `shot_attempt_events_ep`
5. `shot_per_final_third_entry`
6. `own_half_turnovers_ep`
7. representative video quality

## Recommended Execution Order

The best low-cost order is:

1. run `v2`
2. if `v2` looks too reckless, run `v2b_transition`
3. if `v2` still cannot enter danger enough, run `v2c_progression`

If compute allows parallel comparison later, these three presets are the current best small ablation family.

## Final Rule

This ablation is meant to answer:

- does the team need more reward for entering danger?
- more punishment for dangerous turnovers?
- or slightly more credit for line-breaking progress?

That is a much better question than:

- should we just train longer?
