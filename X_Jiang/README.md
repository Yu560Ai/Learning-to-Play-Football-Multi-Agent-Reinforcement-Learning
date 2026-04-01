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
