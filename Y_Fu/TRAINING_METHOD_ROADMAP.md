# Training Method Roadmap

## Goal

Build a stronger `five_vs_five` multi-agent player with a roadmap that has two layers:

1. low-cost methods that are fast to iterate
2. more systematic methods that remain computationally manageable

The idea is not to replace the current path immediately.

The idea is:

- use the cheap path to move fast
- use the more systematic path to reduce guesswork
- let both paths inform each other

## Layer 1: Low-Cost Current Path

These are the methods we should keep using first because they are cheap, fast, and easy to debug.

### 1. Reward engineering + ablation

Purpose:

- quickly identify whether reward shaping is helping or hurting

Why keep it:

- cheapest to change
- easiest to interpret
- best first response when behavior is clearly wrong

### 2. `player_id` input

Purpose:

- reduce homogeneous behavior under a shared policy
- make multi-player coordination easier

Why keep it:

- low-intrusion code change
- much cheaper than changing the full algorithm

### 3. Opponent pool / light self-play

Purpose:

- avoid overfitting to the built-in GRF bot
- keep improving once built-in opponents stop being informative

Why keep it:

- cheaper and more stable than full league training
- useful after the current policy becomes competitive

### 4. Curriculum gate

Purpose:

- stop wasting environment steps on stages that are not passing

Why keep it:

- very high return on almost no extra cost
- avoids spending compute on failed warm-up stages

## Layer 2: More Systematic But Still Manageable

These are the next methods worth exploring once the basic `five_vs_five` line is running.

They are more systematic than hand-tuning everything, but still realistic under limited compute.

### 1. Automatic reward-weight tuning

Basic idea:

- do not learn a fully new reward
- only search or adapt the coefficients of a small shaping family

Examples:

- tune `shot_attempt_reward`
- tune `final_third_entry_reward`
- tune `pass_success_reward`
- tune `turnover_penalty`

Why this is interesting:

- much more systematic than manual weight guessing
- much cheaper than full reward learning

### 2. Population-based training

Basic idea:

- run a small number of training jobs in parallel
- periodically keep the stronger ones
- mutate hyperparameters or reward weights

Why this is interesting:

- naturally fits reward-weight search
- can reuse the same PPO pipeline

Why it is still manageable:

- does not require a brand-new algorithm
- can be done with a small population, not a huge distributed system

### 3. Preference-based reward tuning

Basic idea:

- compare clips or checkpoints
- decide which behavior looks more aligned with real football success
- use that preference to guide reward adjustment

Why this is interesting:

- useful when shaped return disagrees with actual quality
- can make human judgment more explicit and consistent

Why it is still manageable:

- can be done occasionally
- does not require labeling every trajectory

### 4. Learned curriculum

Basic idea:

- use performance signals to decide when to leave `academy_*`
- decide when to switch to `five_vs_five`
- decide when to add harder opponents

Why this is interesting:

- replaces fixed stage timing with performance-driven progression

Why it is still manageable:

- can start with simple rules
- does not require a full meta-RL system

### 5. Meta-gradient reward optimization

Basic idea:

- use an outer objective to update reward coefficients based on downstream performance

Why this is interesting:

- much more systematic than hand-designed shaping

Why it is later-stage:

- significantly more complex
- harder to debug
- easier to get wrong

This should not be the first systematic upgrade.

## Practical Roadmap

## Phase A: Stabilize The Main `five_vs_five` Line

Use:

- current PPO
- parallel environments
- half-day checkpointing
- reward engineering + ablation

Main question:

- can the current setup learn useful `five_vs_five` behavior at all?

## Phase B: Reduce Manual Guesswork

Use:

- reward ablation table
- `player_id`
- opponent pool
- stage pass gates

Main question:

- can we fix the current failures without introducing a heavy new system?

## Phase C: Introduce Small-Scale Systematic Search

Use:

- a small set of reward coefficients
- a small population of runs
- periodic comparison by `win_rate`, `avg_goal_diff`, and videos

Main question:

- can we tune reward and hyperparameters more systematically while keeping compute under control?

## Phase D: Human-Guided Systematic Refinement

Use:

- preference-based comparison between runs or videos
- stronger evaluation filters
- better selection of which reward settings deserve longer training

Main question:

- can we formalize human judgment instead of using informal impressions only?

## What We Should Not Do Yet

Do not jump immediately to:

- full reward learning
- full meta-RL
- large distributed population systems
- complicated self-play leagues

These are expensive and premature unless the simpler route clearly hits a wall.

## Recommended Order

The intended order for this project is:

1. `five_vs_five` main training validation
2. reward engineering + ablation
3. `player_id`
4. opponent pool / light self-play
5. small automatic reward-weight tuning
6. population-based tuning
7. preference-based refinement
8. meta-gradient style reward optimization only if earlier steps justify it

## Why This Roadmap Makes Sense

It balances:

- speed
- interpretability
- engineering cost
- compute cost

The low-cost path gives us momentum.

The more systematic path gives us a way to reduce pure guesswork later without requiring a huge research pivot.
