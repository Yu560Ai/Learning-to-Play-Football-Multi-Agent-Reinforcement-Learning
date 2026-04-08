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

## Additional Academy / SaltyFish-Style Track

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
#
# 2026/4/8

## Final Curriculum Reset And Clean Learning Chain

This section records the most important conclusion reached on `2026/4/8`.

The core lesson was:

- the project was not failing only because of PPO tuning
- it was also failing because the learning chain had become too complicated
- too many experiments, patches, reward variants, and helper rules were mixed into the same path
- as a result, it became hard to tell whether the agent was learning football or only following accidental training biases

After comparing our runs with:

- official Google Research Football academy scenarios
- SaltyFish-style simple training logic
- the project's own failed custom `1v1` iterations

the correct direction became much clearer:

- start from the simplest official academy tasks
- confirm that the agent can really learn there
- then increase difficulty one step at a time
- do not jump directly from empty-goal finishing to goalkeeper + defender pressure
- do not let helper rules become the main source of success

The result is a much cleaner staged curriculum.

## The Correct Step-By-Step Learning Method

The current recommended order is:

1. `academy_empty_goal_close_curriculum`
2. `academy_empty_goal_curriculum`
3. `academy_1v1_close_bridge_fast`
4. `academy_1v1_defended_bridge` (optional harder solo bridge)
5. `academy_pass_and_shoot_curriculum`

`5v5` is no longer treated as the default target. It remains an optional later
experiment only if the solo and two-player stages become stable first.

Current practical next step:

- Stages 1 and 2 are already considered passed.
- The current reliable default entry remains `academy_empty_goal_curriculum`.
- Defended solo stages should be treated as explicit next experiments rather
  than the default launch target until a faster bridge stage is fully
  validated.

### Stage 1: `academy_empty_goal_close_curriculum`

Goal:

- learn the most basic single-player finishing behavior
- move toward the goal
- trigger shot
- convert very easy chances

Why this stage matters:

- it verifies that PPO, observation formatting, action mapping, checkpoint saving, and video export all work
- it removes almost all football complexity except the final act of scoring
- if the agent cannot learn here, the main pipeline is broken

What we observed:

- this stage learns quickly
- success rate can become high
- this confirms that the simplified action set and clean PPO path can learn useful behavior

### Stage 2: `academy_empty_goal_curriculum`

Goal:

- keep the task simple
- but require more self-driven approach and shot timing
- move from very close finishing to a more realistic solo approach

Why this stage matters:

- it forces the policy to do more than stand and shoot
- it tests whether the agent can carry ball progression into a successful finish

Important lesson:

- training rollout metrics may look strong before deterministic video looks strong
- stochastic evaluation and video export are both needed
- the checkpoint can contain real scoring behavior even if deterministic evaluation is still less stable

### Stage 3: `academy_1v1_close_bridge_fast`

Goal:

- introduce exactly one defender
- keep the drill short enough to learn quickly on CPU
- learn to beat one defender and then finish

Why this stage matters:

- this is the first true bridge from empty-goal finishing into real defended play
- it teaches "beat one defender, then finish"
- it is intentionally narrower than official `run_to_score` so that feedback is faster

Most important insight:

- going directly from empty-goal stages into multi-defender official scenarios was too hard
- empty-goal finishing ability did not automatically transfer through full defended pressure at once

So this stage was added as the correct first defended layer.

### Stage 4: `academy_1v1_defended_bridge`

Goal:

- keep solo play but make the defended drill longer and less local
- test whether the agent can sustain defended progression, not just finish a close duel

Why this stage matters:

- this checks whether the agent can generalize from a close duel into a more sustained 1v1 carrying problem
- it is still a solo task, so cooperation has not been introduced yet

Important lesson:

- even here, difficulty can grow too fast if episode length is too long or if possession loss ends the episode immediately
- this stage is optional if the faster bridge already gives stable solo defended finishing

### Stage 5: `academy_pass_and_shoot_curriculum`

Goal:

- only after single-player progression, defended carrying, and finishing are already working
- begin learning two-player coordination
- add pass-to-shot behavior

Why this stage must come last:

- if introduced too early, the agent mixes up three different problems:
  - progression
  - finishing
  - cooperation
- this was one of the main causes of earlier confusion

So the current principle is:

- do not ask for cooperation before stable solo attack exists
- do not ask for `5v5` before stable two-player attack exists

## What Went Wrong Before

The previous failed path had several structural problems:

- custom `1v1` scenario work and official academy work were mixed together
- too many reward variants were added at once
- some helper rules were strong enough to distort behavior
- imitation / prior logic was introduced before the clean curriculum itself was validated
- training conclusions were sometimes drawn from rollout metrics alone without confirming with videos

This created a false picture:

- some runs looked promising in logs
- but the videos showed weak or unstable real behavior

The most important correction was:

- return to official academy tasks first
- prove that the clean training path learns there
- only then step upward

## The Clean Python Logic By File

This section is the main code-level reference for the final report.

### [train.py](/root/Codes/football-rl-win/X_Jiang/train.py)

Responsibility:

- command-line training entry point
- loads a preset from `presets.py`
- applies CLI overrides like device or checkpoint path
- calls the PPO trainer

Meaning in the final chain:

- this is the top-level launch point for every curriculum stage
- it should remain thin
- it should not contain football logic

### [evaluate.py](/root/Codes/football-rl-win/X_Jiang/evaluate.py)

Responsibility:

- load a saved checkpoint
- reconstruct the correct environment and model config
- run deterministic or stochastic evaluation
- optionally export videos

Meaning in the final chain:

- used to confirm whether rollout metrics reflect actual behavior
- videos are mandatory for stage validation
- stochastic evaluation is useful to see whether training rollouts contain real scoring behavior
- deterministic evaluation is useful to see whether the policy has become stable

### [rerender_dump.py](/root/Codes/football-rl-win/X_Jiang/rerender_dump.py)

Responsibility:

- re-render existing `.dump` files into videos without retraining
- useful for fixing visual annotation issues

Meaning in the final chain:

- this is a debugging / reporting utility
- it does not affect learning

### [presets.py](/root/Codes/football-rl-win/X_Jiang/presets.py)

Responsibility:

- defines the training presets
- is the single source of truth for curriculum stages
- controls:
  - environment name
  - action set
  - rollout length
  - timesteps
  - reward shaping weights
  - checkpoint / log directories

Meaning in the final chain:

- this file defines the curriculum itself
- if the curriculum order changes, this file should show it clearly

Current key presets:

- `academy_empty_goal_close_curriculum`
- `academy_empty_goal_curriculum`
- `academy_1v1_close_bridge_fast`
- `academy_1v1_defended_bridge`
- `academy_pass_and_shoot_curriculum`

### [xjiang_football/envs.py](/root/Codes/football-rl-win/X_Jiang/xjiang_football/envs.py)

Responsibility:

- wraps GRF environment creation
- handles action sets
- maps local action ids to GRF actions
- supports video writing and metric extraction
- supports parallel rollout workers

Meaning in the final chain:

- this file is where task interface simplification happens
- the most important clean decision here was to use a very small solo action set:
  - `idle`
  - 8 movement directions
  - `shot`

Why this matters:

- it keeps the learning problem close to the official GRF academy setup
- it avoids noisy action semantics during early solo curriculum

### [xjiang_football/model.py](/root/Codes/football-rl-win/X_Jiang/xjiang_football/model.py)

Responsibility:

- defines the policy and value network
- in the current clean curriculum path, this is a small PPO actor-critic

Meaning in the final chain:

- the current lesson is that model complexity was not the main bottleneck
- the clean curriculum works with a small network

### [xjiang_football/ppo.py](/root/Codes/football-rl-win/X_Jiang/xjiang_football/ppo.py)

Responsibility:

- rollout collection
- GAE advantage computation
- PPO update loop
- checkpoint save / load
- logging
- best-checkpoint tracking

Meaning in the final chain:

- this file should optimize a clean task definition, not compensate for a bad one
- in the final clean curriculum path:
  - PPO is simple
  - no heavy behavior-cloning constraint is active
  - no prior KL is active
  - no rule-based override is used as the main source of success

This is important because earlier versions became too dependent on:

- prior constraints
- helper rules
- overcomplicated shaping

The clean path intentionally avoids that.

### [xjiang_football/rewards.py](/root/Codes/football-rl-win/X_Jiang/xjiang_football/rewards.py)

Responsibility:

- computes reward shaping terms and behavior diagnostics

Meaning in the final chain:

- this file still supports many shaping diagnostics from earlier experiments
- but the clean curriculum intentionally uses only a minimal subset

For the official clean stages, the practical reward logic is mostly:

- GRF scoring reward
- checkpoint-style progression reward
- very weak penalties for obviously bad low-quality shots or regressions

Why this matters:

- this matches the main lesson from SaltyFish and official GRF curriculum usage
- the agent should not be buried under too many shaping terms before basic behaviors are learned

### [xjiang_football/priors.py](/root/Codes/football-rl-win/X_Jiang/xjiang_football/priors.py)

Responsibility:

- defines rule-based single-player priors used for imitation experiments

Meaning in the final chain:

- this file is no longer part of the main clean curriculum path
- it remains useful for future experiments
- but it is not the current recommended default

Reason:

- the project learned that helper priors can easily mask whether the core curriculum actually works

### [imitation_pretrain.py](/root/Codes/football-rl-win/X_Jiang/imitation_pretrain.py)

Responsibility:

- behavior cloning warm-start generation for single-player academy presets

Meaning in the final chain:

- this also remains available
- but it is not part of the current clean official curriculum baseline

Reason:

- first the plain PPO curriculum must be validated
- only then should imitation be reintroduced carefully if needed

### [football-master/gfootball/scenarios/*.py](/root/Codes/football-rl-win/football-master/gfootball/scenarios)

Responsibility:

- official GRF scenario definitions

Meaning in the final chain:

- these are the ground truth tasks for the curriculum
- using them avoids confusion caused by custom scenario side effects

The key official scenarios now used are:

- [academy_empty_goal_close.py](/root/Codes/football-rl-win/football-master/gfootball/scenarios/academy_empty_goal_close.py)
- [academy_empty_goal.py](/root/Codes/football-rl-win/football-master/gfootball/scenarios/academy_empty_goal.py)
- [academy_run_to_score.py](/root/Codes/football-rl-win/football-master/gfootball/scenarios/academy_run_to_score.py)
- [academy_run_to_score_with_keeper.py](/root/Codes/football-rl-win/football-master/gfootball/scenarios/academy_run_to_score_with_keeper.py)
- [academy_pass_and_shoot_with_keeper.py](/root/Codes/football-rl-win/football-master/gfootball/scenarios/academy_pass_and_shoot_with_keeper.py)

## Current Recommended Commands

### Stage 1

```bash
.venv/bin/python X_Jiang/train.py --preset academy_empty_goal_close_curriculum --device cpu
```

### Stage 2

```bash
.venv/bin/python X_Jiang/train.py --preset academy_empty_goal_curriculum --device cpu
```

### Stage 3

```bash
.venv/bin/python X_Jiang/train.py \
  --preset academy_1v1_close_bridge_fast \
  --init-checkpoint X_Jiang/checkpoints/academy_empty_goal_curriculum/best.pt \
  --device cpu
```

### Stage 4

```bash
.venv/bin/python X_Jiang/train.py \
  --preset academy_1v1_defended_bridge \
  --init-checkpoint X_Jiang/checkpoints/academy_1v1_close_bridge_fast/best.pt \
  --device cpu
```

### Stage 5

```bash
.venv/bin/python X_Jiang/train.py \
  --preset academy_pass_and_shoot_curriculum \
  --init-checkpoint X_Jiang/checkpoints/academy_1v1_close_bridge_fast/best.pt \
  --device cpu
```

## What Must Be Checked At Each Stage

Do not rely on one metric only.

The minimum required checks are:

- training logs
- deterministic evaluation
- stochastic evaluation
- video export

The most important learning indicators are:

- `goals_for`
- `success_rate`
- `shots`
- `ball_prog`
- `checkpoints`

Interpretation rule:

- if rollout logs look strong but videos look wrong, the stage is not passed
- if stochastic evaluation works but deterministic evaluation is weak, the behavior exists but is not yet stable
- if deterministic videos clearly show correct task behavior, the stage is passed

## Final Conclusion As Of 2026/4/8

The correct report-level conclusion is:

- the project should not be described as "reward tuning gradually solved football"
- the main breakthrough was rebuilding the learning chain into a clean official curriculum
- the simplest official academy stages were necessary to prove the pipeline could really learn
- the next real challenge is controlled transfer:
  - empty goal
  - defender pressure
  - goalkeeper pressure
  - finally pass-and-shoot cooperation

In short:

- first learn to score
- then learn to score under pressure
- then learn to score together

That is the final correct chain recorded on `2026/4/8`.

## Failure Chain That We Now Reject

The following path is now considered the wrong main path and should not be used as the report's primary story:

- start from custom `1v1`
- add many reward terms at once
- add force-shoot helper rules
- add imitation warm-start before the clean curriculum is validated
- read rollout metrics as if they already prove real football behavior

Why this path failed:

- it mixed too many sources of behavior at once
- it became unclear whether the agent or the helper logic was responsible for success
- videos often contradicted logs
- custom scenarios introduced confounds before official tasks were even solved

This does not mean those experiments were useless.

They were useful because they revealed:

- where the learning chain breaks
- how strong helper rules can distort interpretation
- why official benchmark stages are necessary

But they should now be treated as:

- diagnostic history
- not the clean final method

## Stage-By-Stage Pass Criteria

The curriculum should not move forward just because one metric spikes once.

Each stage must satisfy a practical pass condition.

### Stage 1 Pass Condition: `academy_empty_goal_close_curriculum`

The stage is considered passed when:

- scoring appears quickly and repeatedly
- stochastic video clearly shows repeated direct scoring behavior
- deterministic evaluation is at least partly aligned with the rollout result

What this proves:

- the basic PPO + env + action mapping pipeline is healthy
- the agent can trigger shooting behavior without helper rules

### Stage 2 Pass Condition: `academy_empty_goal_curriculum`

The stage is considered passed when:

- the agent can still score from farther out
- it no longer depends on standing almost on top of goal
- stochastic video contains clear `1-0` episodes

What this proves:

- empty-goal scoring is no longer purely trivial proximity behavior
- the agent can approach and finish

### Stage 3 Pass Condition: `academy_run_to_score_clean`

The stage is considered passed when:

- `goals_for` becomes consistently non-zero
- `success_rate` is not pinned at zero
- the agent does not immediately lose possession to the defender
- videos show forward progression under pressure, not only random shots

What this proves:

- the agent can keep useful attack behavior when a defender is present
- the first defender-pressure transfer step works

### Stage 4 Pass Condition: `academy_run_to_score_official_clean`

The stage is considered passed when:

- the defender-pressure behavior survives with goalkeeper pressure added
- scoring remains non-zero after transfer from Stage 3
- the video shows actual finishing attempts that beat the keeper, not just noise

What this proves:

- the agent has moved from empty-goal finishing to real defended finishing

### Stage 5 Pass Condition: `academy_pass_and_shoot_with_keeper`

The stage is considered passed when:

- pass behavior actually appears
- goals come from meaningful pass-to-shot sequences
- the agent does not collapse back into isolated solo random shots

What this proves:

- solo attack ability is strong enough to support simple cooperation

## What Has Already Been Verified

As of `2026/4/8`, the following statements are already supported by actual runs:

1. `academy_empty_goal_close_curriculum` can be learned cleanly with plain PPO.
2. `academy_empty_goal_curriculum` can also produce real scoring episodes.
3. Strong rollout metrics alone are not enough; stochastic and deterministic videos must also be checked.
4. Jumping directly from empty-goal stages to `academy_run_to_score_with_keeper` is too large a transfer step.
5. A missing middle layer between empty-goal scoring and goalkeeper-pressure scoring is required.

These are not guesses anymore.

They are conclusions backed by:

- training logs
- saved checkpoints
- deterministic evaluation
- stochastic evaluation
- exported videos

## What Is Still Open

The following questions remain open and should be treated as current research work rather than solved facts:

1. How stable will `academy_run_to_score_clean` become after enough updates?
2. How much of that behavior will transfer into `academy_run_to_score_official_clean`?
3. At what point is the policy stable enough to begin `academy_pass_and_shoot_with_keeper`?
4. How much of the academy-learned attack behavior can be transferred back into `5v5`?

These questions should be evaluated in this order.

Do not jump to later questions before earlier ones are clearly answered.

## Clean Interpretation Rules For Future Reporting

When writing the final report, the following interpretation rules should be followed strictly.

### Rule 1

Do not claim a task is solved only because rollout `success_rate` is high.

Why:

- rollout statistics can be inflated by stochastic action sampling
- videos may still show unstable or poor deterministic behavior

### Rule 2

Do not claim helper logic taught football unless the same behavior survives after helper logic is removed.

Why:

- force-shoot and related helpers can create fake progress
- the report must distinguish:
  - agent-learned behavior
  - helper-induced behavior

### Rule 3

If deterministic video and stochastic video disagree, write that explicitly.

Interpretation:

- stochastic success means the policy has discovered the behavior distributionally
- deterministic failure means the policy has not yet stabilized around it

### Rule 4

If a new curriculum stage fails after a previous one succeeded, do not immediately conclude that the previous stage was useless.

Interpretation:

- usually this means the transfer jump was too large
- not that the previous skill was fake

### Rule 5

Prefer official GRF curriculum evidence over custom scenario evidence when establishing the main report narrative.

Why:

- official scenarios are standardized
- custom scenarios are better treated as later experiments

## Current Recommended Narrative For The Final Report

The report should now describe the project as:

- a curriculum reconstruction effort
- not a reward-hacking success story
- not a pure hyperparameter tuning story

The clean narrative should be:

1. The original complex path mixed too many learning sources.
2. Official academy stages were used to re-establish a trustworthy baseline.
3. The agent first learned empty-goal finishing.
4. Transfer to defended finishing required inserting a defender-only intermediate stage.
5. Only after defended solo finishing is stable should cooperative pass-and-shoot be learned.

This is the clearest and most defensible account of what the code now represents.

## Relationship To SaltyFish And GRF Official Design

The final `2026/4/8` chain should be described as aligned with two important ideas:

### From official GRF academy

- learn through progressively harder scenarios
- validate on standardized tasks
- do not hide environment difficulty behind custom task assumptions too early

### From SaltyFish

- keep the learning problem clean
- avoid piling on too many simultaneous behavior objectives
- use simple, interpretable reward structure first

The final clean path therefore borrows:

- official scenario progression from GRF
- problem simplification discipline from SaltyFish

## Current Recommended Practical Workflow

Until this line is fully completed, the practical workflow should be:

1. Train one curriculum stage.
2. Save `best.pt`.
3. Run deterministic evaluation.
4. Run stochastic evaluation.
5. Export video.
6. Decide pass / fail.
7. Only then move to the next stage.

This is now the correct operational habit for the project.

## Final Status Summary On 2026/4/8

At the end of this revision, the state of the project is:

- the clean official academy path is working
- the first two empty-goal curriculum stages have real evidence of success
- direct transfer to keeper pressure was shown to be too difficult
- a defender-only intermediate stage was introduced as the next correct step
- the README now reflects the correct learning chain and should be used as the main report reference
