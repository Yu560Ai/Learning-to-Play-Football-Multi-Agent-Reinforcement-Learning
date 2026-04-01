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
