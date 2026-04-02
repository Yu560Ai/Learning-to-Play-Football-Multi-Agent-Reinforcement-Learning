# X_Jiang
# 2026.4.1

## Overview

This directory contains X_Jiang's Google Research Football reinforcement learning pipeline for the `5_vs_5` setting.

The current version focuses on a role-aware multi-agent PPO setup for five controllable players:

- `GK` goalkeeper
- `LD` left defender
- `RD` right defender
- `CM` central midfielder / pivot
- `ST` striker

The goal of this version is to move beyond a naive shared-policy setup and improve:

- goalkeeper stability near goal
- defensive ball pressure from outfield players
- better left/right defensive structure
- more active midfield support behavior
- clearer attacking progression and passing support

This implementation keeps the existing local Google Research Football workflow:

- Python 3.12 compatible
- local `football-master` source tree
- local venv workflow
- no Docker
- no simulator replacement

## Current Model Design

### 1. Environment Layer

Main file:

- [envs.py](/root/Codes/football-rl-win/X_Jiang/xjiang_football/envs.py)

The environment wrapper:

- reuses the local `football-master` path patching logic
- creates the GRF environment through `gfootball.env.create_environment`
- supports `5_vs_5`
- supports configurable number of controlled left-side players
- supports video export
- formats observations, rewards, and actions for PPO
- exposes score and raw observation helpers

This version assumes a five-player controllable setup for `5_vs_5`, including the goalkeeper.

### 2. PPO Training Pipeline

Main files:

- [train.py](/root/Codes/football-rl-win/X_Jiang/train.py)
- [ppo.py](/root/Codes/football-rl-win/X_Jiang/xjiang_football/ppo.py)
- [presets.py](/root/Codes/football-rl-win/X_Jiang/presets.py)

Training uses PPO with:

- rollout collection
- generalized advantage estimation
- minibatch PPO updates
- checkpoint save / load
- progress logging

This version is designed to remain runnable on CPU for local iteration.

### 3. Role-Aware Observation Design

Main idea:

- keep GRF environment observations
- add tactical structure through role-aware reasoning

The intended five tactical roles are:

1. `GK`
2. `LD`
3. `RD`
4. `CM`
5. `ST`

The role design is meant to encourage:

- a deeper, more stable goalkeeper
- left/right defenders that protect their channels
- a midfielder that connects play and supports possession
- a striker that stays more advanced and applies first-line pressure

### 4. Reward Shaping

Main file:

- [rewards.py](/root/Codes/football-rl-win/X_Jiang/xjiang_football/rewards.py)

This version uses modular reward shaping on top of GRF rewards.

Current reward design includes behavior guidance for:

- defensive ball chasing
- goalkeeper positioning discipline
- defensive recovery
- progression and possession-related shaping

The shaping is intended to guide behavior without replacing the main football objective.

### 5. Evaluation

Main file:

- [evaluate.py](/root/Codes/football-rl-win/X_Jiang/evaluate.py)

Evaluation supports:

- checkpoint loading
- deterministic or stochastic rollout
- rendering or no-render evaluation
- video export
- episode summary logging

## File Structure

- [train.py](/root/Codes/football-rl-win/X_Jiang/train.py): training entry point
- [evaluate.py](/root/Codes/football-rl-win/X_Jiang/evaluate.py): evaluation entry point
- [presets.py](/root/Codes/football-rl-win/X_Jiang/presets.py): preset configurations
- [xjiang_football/envs.py](/root/Codes/football-rl-win/X_Jiang/xjiang_football/envs.py): environment wrapper
- [xjiang_football/ppo.py](/root/Codes/football-rl-win/X_Jiang/xjiang_football/ppo.py): PPO trainer
- [xjiang_football/rewards.py](/root/Codes/football-rl-win/X_Jiang/xjiang_football/rewards.py): reward shaping

## Current Training / Evaluation Workflow

### Train

Example:

```bash
python X_Jiang/train.py --preset five_v_five_debug
```

### Evaluate

Example:

```bash
python X_Jiang/evaluate.py --checkpoint X_Jiang/checkpoints/five_v_five_debug/update_40.pt --episodes 1 --save-video
```

## Current Shared Model

A curated model snapshot can be stored under:

- `best_models/X_Jiang/`

Large local training checkpoints under:

- `X_Jiang/checkpoints/`

should remain local and should not be committed directly unless moved into `best_models/` as a selected shared artifact.

## Current Status

This version is a stronger experimental upgrade over the earlier simple shared-policy baseline, but it is still an in-progress research model rather than a final polished result.

At the current stage:

- the code can train and evaluate locally
- role-aware training is the main direction
- goalkeeper behavior is more constrained than before
- outfield behavior still needs further improvement and stabilization
- `five_vs_five` remains the main project focus

## Known Limitations

- short training runs are not enough to produce stable team play
- outfield coordination is still inconsistent
- defensive pressure and attacking support still need tuning
- reward shaping and presets likely need further iteration

## Next Priorities

1. Improve defender pressure and recovery behavior
2. Improve midfielder support and circulation behavior
3. Improve striker positioning and first-line press behavior
4. Run longer and more stable `five_vs_five` training
5. Compare checkpoints using consistent evaluation metrics and videos

## Notes For GitHub Upload

Recommended to commit:

- Python source code
- scenario/config changes
- selected shared model under `best_models/X_Jiang/`
- documentation updates

Do not commit:

- local training checkpoints in `X_Jiang/checkpoints/`
- replay videos
- dumps
- logs
- local virtual environments

## Strategy

### Current RL Training Strategy

At the current stage, the main reinforcement learning strategy for `X_Jiang` is:

- keep PPO as the core optimization method
- treat each controlled player as an individual training unit inside the same football episode
- let each player learn from its own observation, action, reward, and advantage signal
- keep all players inside one shared team environment rather than splitting the match into separate environments

In practical terms, the training logic is:

1. Start one `5_vs_5` GRF episode
2. Control the left-side players in the same match
3. For each timestep, collect per-player:
   - observation
   - chosen action
   - action log-probability
   - value estimate
   - reward
4. Store rollout data across time
5. Use PPO + GAE to compute advantages
6. Update the policy from the collected rollout

### Why We Chose This Direction

The reason for this strategy is that football is not a single-agent problem.

Even though the team shares one match environment, the players do not have the same job:

- goalkeeper should stay deep and protect goal
- defenders should pressure and recover in their own channels
- midfielder should connect play and support both sides of the ball
- striker should stay more advanced and provide first-line pressure

If we train everything as one completely undifferentiated behavior, the most common failure mode is:

- goalkeeper learns something partially useful
- outfield players become too passive
- players do not pressure the ball correctly
- off-ball support shape is poor

So the current strategy is to make PPO learn player behavior at the per-player level, while still learning inside one full team game.

### Per-Player PPO Logic

The intended logic is:

- each player produces its own action at each timestep
- each player receives reward feedback from the same football transition
- PPO optimization is done over the aggregated batch of player-timestep samples
- role-aware reward shaping is used so that different players are pushed toward different tactical behavior

This means we are not training one giant monolithic "team action" directly.
Instead, we are training player decisions within a shared team rollout.

Conceptually:

- same match
- multiple controlled players
- PPO updates built from player-wise samples

### Shared Team Context + Role Separation

Our current research direction is not five fully isolated independent projects with five completely unrelated training pipelines.

Instead, the logic is:

- players act separately
- players are evaluated inside the same match context
- reward shaping can emphasize different tactical responsibilities
- the whole system should still improve as one coordinated team

So the target is:

- tactical separation at the player level
- coordination at the team level
- PPO as the common optimization framework

### What "Separate Training" Means Here

When we say the players are "trained separately", we do **not** mean:

- launching five unrelated matches
- using five unrelated football simulators
- building a completely different environment stack

What we mean is:

- each player is treated as its own decision-making unit
- the rollout buffer contains player-wise samples
- the learning signal should allow different players to specialize
- specialization is guided through role design and reward shaping

So the separation is at the policy-learning and behavior-learning level, not at the environment level.

### Current Practical Goal

The immediate practical goal of this strategy is:

- goalkeeper remains stable near goal
- nearest outfield player actively chases the ball when defending
- other outfield players recover into useful support positions
- defenders hold left/right structure better
- midfielder becomes a connector rather than a spectator
- striker stays more advanced and contributes to pressure

### Next Strategy Iteration

If longer PPO training still shows weak ball pressure from outfield players, the next iteration should continue along this same strategy direction:

- stronger per-player defensive chase incentives
- clearer role-conditioned behavior
- better distinction between first presser and covering teammates
- more stable long-run PPO training and evaluation

In short, the current strategy is:

- multi-player PPO
- player-wise learning signals
- shared team environment
- role-guided specialization
- football coordination learned inside one match rollout
# 2026.4.2
## Overview of new update

Today the project moved from incremental reward patching toward a more structural attacking curriculum design.

The main change is that the training setup no longer treats the current `5_vs_5` problem as something that should be learned only from full-match PPO rollouts. After several failed long runs, the codebase now explicitly acknowledges that the model was learning defensive structure and low-risk local behaviors much faster than it was learning actual attack construction.

### 1. Current Control Assumption

The current `5_vs_5` setup trains **4 outfield players only**.

This is important:

- the goalkeeper is **not** part of the learned policy
- the goalkeeper is therefore no longer treated as a normal attacking reset target
- any logic that assumes "safe reset to goalkeeper" is now considered structurally dangerous

This matches the practical GRF `5_vs_5` control pattern more closely and avoids the previous own-goal / unsafe-reset failure mode.

### 2. Official GRF Academy Curriculum Is Now Part of the Design

Instead of inventing a fake teaching environment, the new pipeline now directly uses official GRF academy scenarios through `gfootball.env.create_environment(...)`.

New curriculum-oriented presets were added:

- `academy_pass_and_shoot_attack_support`
- `academy_3v1_attack_support`

These are built on top of official GRF scenarios:

- `academy_pass_and_shoot_with_keeper`
- `academy_3_vs_1_with_keeper`

The reason for this change is simple:

- full `5_vs_5` was teaching defensive shape faster than attack
- the agent needed a narrower learning path for:
  - pass selection
  - support positioning
  - shot triggering
  - short attacking sequences

This idea is directly inspired by the older SaltyFish-style learning path, but adapted to our current multi-player tactical framework rather than copied literally.

### 3. Support Positioning Was Reworked

One of the biggest structural problems in earlier versions was that off-ball support positions were often only "geometrically reasonable" rather than actually useful for attack continuation.

To address this, support behavior is no longer based only on static role targets.

A new dynamic attacking support target generator was added:

- it uses the current ball position
- it uses the current carrier position
- it uses the supporting player's role and lane
- it tries to place support in a more useful forward or diagonal passing lane
- it avoids collapsing onto the same teammate lane
- it tries to avoid nearby opponent occupation

This is the first step toward changing support from:

- "shape maintenance"

into:

- "real attacking passing network support"

### 4. Critic Structure Was Upgraded

Another major change is that the value function is no longer forced to stay fully shared across all tactical modes.

The model now supports:

- specialized policy heads
- specialized value heads

This matters because:

- `on_ball`
- `support_attack`
- `first_defender`
- `support_cover`

do not have the same return structure.

In earlier versions, the shared critic likely encouraged overly averaged and conservative solutions. The new mode-aware value heads are meant to reduce that problem.

### 5. Reward Logic Was Not Removed, But Reinterpreted

Reward shaping is still present, but the project direction has changed.

The new working assumption is:

- reward shaping alone cannot fix the model
- the model needs the correct learning decomposition first
- shaping should support that decomposition rather than compensate for a bad one

Recent adjustments therefore focused on:

- reducing meaningless safe resets
- discouraging support behind the ball
- encouraging more useful forward attacking support
- keeping anti-stall / anti-loop constraints

At the same time, adaptive reward scaling was added so that attacking incentives can recover if training starts collapsing into defensive local optima.

### 6. Current Training Status

The model is **not solved yet**.

What has improved:

- goalkeeper-reset abuse has been reduced
- some attacking academy runs now stay in attack-oriented modes
- the codebase is now structurally closer to a staged learning pipeline than before
- support positioning logic is more attack-aware than in earlier versions

What is still wrong:

- academy training can still collapse into scripted local attack patterns
- some runs overuse `shoot_ball` without actually converting attacks
- some runs still fail to create stable pass-to-shot behavior
- full `5_vs_5` play is still not consistently producing real football coordination

So the current state is best described as:

- no longer just patching symptoms
- beginning a real framework transition
- still in the middle of attack-system reconstruction

### 7. Current Research Direction

The current direction after today's update is:

1. use official GRF academy stages to learn attacking sub-skills first
2. keep the goalkeeper outside the learned attacking logic in `5_vs_5`
3. treat support players as attacking network nodes rather than static geometry points
4. keep mode-based tactical routing
5. use specialized value heads to reduce conservative averaging
6. transfer the stronger academy attack behavior back into `5_vs_5`

In short, today's update marks a shift from:

- fixing bad behaviors one by one

to:

- rebuilding the learning path so the agent can actually learn useful football content

## 2026.4.2 Additional Academy / SaltyFish-Style Track

After further testing, the staged tactical academy path still showed a clear problem:

- the agent could improve some shaped signals
- but it still failed to convert that into reliable goals or wins

One representative issue was that the agent often learned to:

- overuse `shoot_ball`
- shoot without real possession quality
- finish many updates without meaningful episode success statistics

So a second experimental track was added instead of continuing to only tune the tactical multi-agent academy presets.

### Why This New Track Was Added

The main conclusion from the failed academy tactical runs was:

- the tactical action space is still relatively large
- reward shaping is still easy to exploit
- staged academy learning in the tactical framework is not yet simple enough

Because of that, a new baseline was created based on the core SaltyFish idea:

- simpler PPO setup
- `simple115v2` observations
- reduced low-level action set
- lighter reward shaping
- more direct football behavior signals

But this was **not** implemented as a compatibility wrapper around the older `11_vs_11` SaltyFish code.

Instead, a new version was written directly for this project's own `5_vs_5` setting.

### New 5v5 SaltyFish-Style Baseline

New files:

- [train_saltyfish.py](/root/Codes/football-rl-win/X_Jiang/train_saltyfish.py)
- [evaluate_saltyfish.py](/root/Codes/football-rl-win/X_Jiang/evaluate_saltyfish.py)
- [saltyfish_five_baseline/ppo.py](/root/Codes/football-rl-win/X_Jiang/saltyfish_five_baseline/ppo.py)
- [saltyfish_five_baseline/env.py](/root/Codes/football-rl-win/X_Jiang/saltyfish_five_baseline/env.py)
- [saltyfish_five_baseline/features.py](/root/Codes/football-rl-win/X_Jiang/saltyfish_five_baseline/features.py)
- [saltyfish_five_baseline/model.py](/root/Codes/football-rl-win/X_Jiang/saltyfish_five_baseline/model.py)
- [saltyfish_five_baseline/evaluate.py](/root/Codes/football-rl-win/X_Jiang/saltyfish_five_baseline/evaluate.py)

This new branch is designed specifically for:

- `5_vs_5`
- `4` controlled outfield players
- reduced GRF action space
- grouped `simple115v2` feature processing
- optional engineered features built for `5v5`, not `11v11`

### Key Design Choices

The purpose of this branch is not to replace the main tactical framework immediately.

Its purpose is to answer a narrower question first:

- can a simpler SaltyFish-style learning setup produce meaningful learning signal in our `5_vs_5` environment faster than the tactical academy branch?

Important design choices:

- keep PPO hyperparameters close to the SaltyFish-inspired setup
- keep the observation format simple
- reduce the action space to a smaller football-relevant subset
- use direct possession / pass / carry / territory / shot shaping
- avoid tactical-mode routing in this branch

### Current Commands

Quick training run:

```bash
python X_Jiang/train_saltyfish.py --preset five_v_five_quick
```

Longer baseline training:

```bash
python X_Jiang/train_saltyfish.py --preset five_v_five_base
```

Evaluation:

```bash
python X_Jiang/evaluate_saltyfish.py \
  --checkpoint X_Jiang/checkpoints/saltyfish_five_v_five/latest.pt \
  --episodes 10 \
  --compare-random \
  --deterministic
```

### What Was Learned From The First Runs

Early runs on this new branch immediately showed a useful diagnostic difference:

- the logging became much easier to interpret
- it became more obvious when the agent was spamming shots
- it became easier to see whether the agent actually had the ball when shooting

One early pattern was:

- many shot attempts
- near-zero `shots_on_ball`
- near-zero pass completions
- almost no attacking-third possession

This suggests that the first weak point in the new branch is not only PPO tuning.

It is also:

- bad action preference calibration
- insufficient penalty for meaningless shots
- insufficient reward pressure toward real possession progression

### Additional Logging Added

To make diagnosis easier, the new branch logs:

- `goals_for`
- `goals_against`
- `success_rate`
- `score_examples`
- `mean_steps_left`
- `min_steps_left`

These were added because some early runs produced:

- no completed episodes inside a rollout window
- unclear score visibility
- ambiguity about whether the environment had actually reached terminal states

So this branch now explicitly tracks both:

- football outcomes
- episode progression / termination behavior

### Current Interpretation

The current interpretation is:

- the tactical academy branch is still useful as a research direction
- but it is not yet the fastest path to confirming whether the project can learn useful `5_vs_5` football behavior
- the new SaltyFish-style `5v5` branch is now the simpler diagnostic baseline

At this stage, the practical workflow is:

1. use the SaltyFish-style `5v5` branch to validate that learning trends are real
2. inspect whether possession, passing, and shot quality improve
3. only then decide whether to transfer ideas back into the tactical multi-agent branch

So this new branch should be treated as:

- a simpler control experiment
- a debugging baseline
- a lower-complexity reference for future `5_vs_5` training work
