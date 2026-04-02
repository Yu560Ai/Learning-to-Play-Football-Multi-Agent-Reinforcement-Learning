# Five-v-Five Reward V2 Plan

## Purpose

This note defines the next concrete `five_vs_five` reward preset for the `Y_Fu` PPO line:

- `five_vs_five_reward_v2`

The aim is not to add more reward terms.

The aim is to make the reward:

- more aligned with real football process quality
- less aligned with harmless local behavior
- harder to exploit with empty possession or direction-heavy drift

## Main Design Rule

The reward should answer a coach-like question:

- did this possession become more dangerous in a meaningful way?

and not mainly:

- did the team keep the ball?
- did the team move forward a little?
- did a pass technically complete?

So the reward is centered on:

1. entering danger
2. converting danger into shots
3. avoiding the clearest dangerous turnover

## The Preset

The new preset lives in [ppo.py](/home/yuhan/Codes/RL/Learning-to-Play-Football-Multi-Agent-Reinforcement-Learning/Y_Fu/yfu_football/ppo.py) as:

- `five_vs_five_reward_v2`

Key reward coefficients:

- `pass_success_reward = 0.02`
- `pass_failure_penalty = 0.02`
- `pass_progress_reward_scale = 0.02`
- `shot_attempt_reward = 0.08`
- `attacking_possession_reward = 0.0`
- `final_third_entry_reward = 0.08`
- `possession_retention_reward = 0.0`
- `possession_recovery_reward = 0.01`
- `defensive_third_recovery_reward = 0.015`
- `opponent_attacking_possession_penalty = 0.0`
- `own_half_turnover_penalty = 0.02`

## Why These Values

### 1. Pass Reward Is Kept Small

Passing is important, but pass completion is not the true target.

If pass reward is too large, PPO can learn:

- harmless circulation
- shallow forward passing
- pass-first behavior without real chance creation

So in reward v2:

- pass success still matters
- but it is clearly weaker than the danger-creation terms

### 2. Shot And Final-Third Entry Are The Main Dense Attack Terms

These are the best available process proxies in the current codebase.

They are closer to real football attacking quality than:

- generic possession
- retention
- pass completion alone

This is why both are set to `0.08`.

The design intent is:

- entering danger is good
- turning danger into a shot is also good

This creates a better process chain than generic progression reward.

### 3. Generic Possession Reward Stays At Zero

These terms remain disabled:

- `attacking_possession_reward`
- `possession_retention_reward`

Reason:

- they are too frequent
- they are too easy to game
- they let the policy collect reward without becoming threatening

This was one of the clearest failure modes in the earlier PPO line.

### 4. Transition Discipline Is Kept, But Only In A Clear Form

The clearest defensive-discipline term is:

- `own_half_turnover_penalty`

This stays active and is slightly strengthened relative to the earlier reward-only revision.

Reason:

- losing the ball in your own half is a real football mistake
- it is more attributable than broad team-level defensive blame

At the same time, this stays disabled:

- `opponent_attacking_possession_penalty`

because it is noisier and less attributable.

### 5. Recovery Terms Stay Small

The recovery rewards remain small:

- `possession_recovery_reward = 0.01`
- `defensive_third_recovery_reward = 0.015`

These are useful, but should not dominate.

The goal is:

- do not train a "recover and survive" policy
- train a policy that recovers and then builds a real attack

## Why This Is More Realistic

Real football training does not usually reward:

- possession for its own sake
- pass completion for its own sake
- merely being in the attacking half

It values:

- entries into dangerous zones
- turning entries into shots
- avoiding self-destructive turnovers
- recovering the ball in useful areas

Reward v2 is a better approximation of that coaching logic than the earlier reward family.

## How To Judge Reward V2

Do not judge it by shaped return first.

Judge it by these metrics together:

- `goals_for`
- `avg_goal_diff`
- `final_third_entries_ep`
- `shot_attempt_events_ep`
- `shot_per_final_third_entry`
- `own_half_turnovers_ep`
- `opponent_dangerous_possessions_ep`
- representative video

The key question is:

- does reward v2 increase attack completion quality without exploding transition mistakes?

## Expected Good Signs

Good early signs would be:

- `final_third_entries_ep` increases
- `shot_attempt_events_ep` increases
- `shot_per_final_third_entry` improves
- `goals_for` is no longer pinned at zero
- matches stop being empty full-length losses as often

## Expected Bad Signs

Bad signs would be:

- `final_third_entries_ep` increases but `shot_attempt_events_ep` stays flat
- `shot_attempt_events_ep` rises but goals remain flat and videos look forced
- `own_half_turnovers_ep` rises sharply
- the policy becomes reckless rather than purposeful

If that happens, reward v2 is still not the right balance.

## Recommended First Use

The cleanest first test is:

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

If you want the structural help from role information too, use:

```bash
python Y_Fu/train.py \
  --preset five_vs_five_reward_v2 \
  --device cpu \
  --use-player-id \
  --num-envs 4 \
  --rollout-steps 256 \
  --update-epochs 4 \
  --num-minibatches 1 \
  --total-timesteps 2000000 \
  --init-checkpoint Y_Fu/checkpoints/academy_pass_and_shoot_with_keeper/update_10.pt \
  --seed 42
```

The first run should still be treated as an early transfer check, not a full overnight commitment.

## Final Rule

Reward v2 should be thought of as:

- less reward for "football-like motion"
- more reward for "dangerous attack completion plus transition discipline"

That is the most realistic next step available within the current codebase.
