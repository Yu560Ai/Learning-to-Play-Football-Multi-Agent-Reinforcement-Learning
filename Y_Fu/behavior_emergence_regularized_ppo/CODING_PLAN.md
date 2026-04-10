# Behavior Emergence Regularized PPO Coding Plan

## Purpose

This document converts the adjusted paper outline into a concrete coding plan for the current `Y_Fu/` training branch.

The target is not a full algorithm rewrite. The target is a controlled new research line with:

- a 2-agent attacking GRF setup
- shared-policy PPO as the fixed backbone
- optional regularization terms added one at a time
- evaluation focused on when cooperative behaviors emerge

## Working Assumption

This plan assumes the new work should live under:

- `Y_Fu/behavior_emergence_regularized_ppo/`

and should reuse the existing `Y_Fu/yfu_football/` PPO code rather than starting a separate trainer from scratch.

## Current Repo Baseline

The current code already provides useful pieces:

- `Y_Fu/yfu_football/ppo.py`: shared-policy PPO trainer, presets, logging, checkpointing
- `Y_Fu/yfu_football/envs.py`: GRF wrapper, reward shaping hooks, action sanitization
- `Y_Fu/yfu_football/model.py`: CNN and MLP actor-critic models
- `Y_Fu/evaluate.py` and `Y_Fu/evaluate_multiagent.py`: evaluation entry points

The main mismatch with the new paper direction is:

- current presets are still centered on `extracted` observations
- the current action interface is the native GRF discrete space, not the paper's reduced action space
- regularization is not yet exposed as a first-class PPO objective component
- behavior probing is only partly implemented through training diagnostics

## Spec To Implement

### Environment Target

- scenario focus: 2-agent attacking stage
- learned players: 2 attacking agents
- non-learned environment actors: goalkeeper and defenders
- first preferred scenario: `academy_pass_and_shoot_with_keeper`

If the built-in Academy scenario cannot exactly match the paper assumptions, create a custom wrapper first and postpone a custom scenario file until it is actually necessary.

### Observation Target

Implement a player-centric vector observation:

- target size: `55`
- one local observation per controlled agent
- shared policy, decentralized execution

Feature groups from the paper:

- self features
- teammate features
- opponent 1 features
- opponent 2 features
- goalkeeper features
- ball features
- context features
- agent identity features

### Action Target

Implement a reduced discrete action space for this line only.

Paper target currently says:

- `|U| = 14`
- 8 movement directions
- short pass
- long pass
- shot
- sprint
- dribble

Important spec issue:

- this enumeration only totals `13`, not `14`

Before coding the action mapper, freeze the exact list. Most likely candidates for the missing action are:

- `idle`
- `release_direction`
- `high_pass`

Do not code this ambiguously. Lock the final list first and keep it fixed for all experiments in this line.

## Implementation Strategy

### Phase 1: Minimal Runnable Baseline

Goal:

- run PPO in the 2-agent attacking setup with no new regularizers

Tasks:

- add a dedicated preset for the new line in `Y_Fu/yfu_football/ppo.py`
- keep shared-policy PPO unchanged as the optimization backbone
- verify rollout collection, return computation, PPO update, save/load, and evaluation
- confirm the scenario truly runs with 2 controlled attackers

Exit condition:

- training runs end-to-end
- checkpoints load
- evaluation videos can be dumped

### Phase 2: Reduced Observation and Action Interfaces

Goal:

- match the new paper setup rather than the current generic GRF interface

Tasks:

- add a custom observation builder in `Y_Fu/yfu_football/envs.py`
- derive the 55-d vector from raw GRF observations
- add a reduced-action mapper from local action ids to GRF action ids
- keep invalid-action filtering for no-ball players
- switch the model default for this line to an MLP or residual MLP, not CNN

Exit condition:

- env wrapper emits `[num_players, 55]`
- policy outputs only the reduced action set
- evaluation confirms actions map correctly in-game

### Phase 3: Regularized PPO Objective

Goal:

- support controlled additions to PPO without changing the training backbone

Priority order:

1. entropy coefficient scheduling or explicit entropy tracking cleanup
2. KL regularization to a frozen prior policy
3. BC-style auxiliary loss from demonstrations or heuristic labels

Tasks:

- extend `PPOConfig` with regularization weights and stage flags
- log each loss term separately
- make every extra term individually switchable
- support loading a reference policy for KL regularization
- avoid mixing multiple new terms before each one has a standalone baseline

Exit condition:

- `L_total = L_ppo + lambda_kl * L_kl + lambda_bc * L_bc + lambda_ent * L_entropy`
- each extra term can be enabled or disabled from CLI or preset
- training logs show per-term magnitudes

### Phase 4: Evaluation for Behavior Emergence

Goal:

- measure more than just reward

Tasks:

- keep episode return and goal success rate
- add probe-style evaluation scripts for:
  - ball progression
  - passing and coordination
  - finishing under goalkeeper pressure
- track action frequencies over training
- track cooperation metrics such as:
  - pass-to-shot conversion
  - pass success rate
  - coordinated attack success rate
- save summary tables by checkpoint

Exit condition:

- one checkpoint can be evaluated on both task success and behavior probes
- emergence curves can be plotted from saved logs

### Phase 5: Human-Guided Refinement Loop

Goal:

- make the paper's iterative refinement claim executable

Tasks:

- define a stage log for each intervention
- record why a new bias was added
- keep one change per experiment whenever possible
- compare against the previous stage checkpoint, not only against random

Exit condition:

- each refinement has a written reason, code diff, and evaluation result

## Planned File Ownership

Primary files expected to change first:

- `Y_Fu/yfu_football/envs.py`
- `Y_Fu/yfu_football/ppo.py`
- `Y_Fu/yfu_football/model.py`
- `Y_Fu/evaluate.py`
- `Y_Fu/evaluate_multiagent.py`

Likely new files:

- `Y_Fu/behavior_emergence_regularized_ppo/EXPERIMENT_LOG.md`
- `Y_Fu/behavior_emergence_regularized_ppo/PROBE_TASKS.md`
- `Y_Fu/behavior_emergence_regularized_ppo/METRICS_SPEC.md`
- `Y_Fu/yfu_football/regularizers.py`

## First Coding Order

Implement in this order:

1. freeze the exact reduced action set
2. add a new preset for the 2-agent line
3. build the 55-d observation adapter
4. swap this line to a vector-policy backbone
5. run PPO without KL or BC
6. add evaluation probes and behavior metrics
7. add KL regularization
8. add BC regularization only after data or heuristic labels are clearly defined

## Practical Constraints

### Keep Fixed

- PPO remains the backbone
- parameter sharing remains the default actor setup
- environment changes should be isolated in the wrapper first

### Avoid Early

- switching to a completely new RL algorithm
- mixing reward shaping, KL, BC, and observation redesign all at once
- moving to `5_vs_5` before the 2-agent stage is behaviorally clear

## Open Decisions To Resolve Before Main Coding

These should be fixed explicitly in the spec before deeper implementation:

1. What is the 14th reduced action?
2. Is the target scenario exactly `academy_pass_and_shoot_with_keeper`, or a custom variant of it?
3. Which exact raw-observation fields will define the 55-d vector?
4. What is the source of BC supervision: demonstrations, scripted heuristics, or curated rollouts?
5. Is KL regularization toward the immediately previous checkpoint, or a separately selected prior policy?

## Definition Of Done For The New Line

This coding line is in a good first state when all of the following are true:

- the new preset trains in the 2-agent attacking scenario
- observations are the intended vector form
- actions are the frozen reduced set
- PPO can run with and without regularizers
- evaluation reports both task performance and cooperation metrics
- experiment history is documented inside this folder

## Immediate Next Task

The next implementation step should be:

- freeze the reduced action list and write the exact 55-d observation schema before editing trainer logic further

Without that, the code will drift away from the paper spec again.
