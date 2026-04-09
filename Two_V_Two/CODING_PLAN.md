# 2v2 Coding Plan

## Scope

This folder is the shared workspace for the new `2v2` direction.

We treat this line as a fresh project goal:

- focus on `2v2` cooperative football
- use a simplified attacking setting
- study passing, positioning, and finishing
- keep the implementation modular so regularized PPO can be added in stages

## Main Objective

Build a clean training and evaluation pipeline for a `2v2` cooperative football task in Google Research Football, then use it to analyze how cooperative behavior emerges over training.

## Initial Design

### Environment

Target setup:

- `2` controlled attacking agents
- environment-controlled goalkeeper
- environment-controlled defenders
- one shared team reward

The first implementation should prefer the simplest scenario that already matches most of this setup. If no built-in scenario matches well enough, define a custom wrapper or scenario after the baseline is runnable.

Current baseline choice:

- built-in scenario: `academy_run_pass_and_shoot_with_keeper`
- controlled left-side players: `2`
- built-in goalkeeper and defenders remain environment-controlled

### Learning Setup

Base training method:

- shared-policy PPO
- decentralized execution
- one policy used by both controlled agents

Regularization and extra losses should be added only after the plain PPO baseline is stable.

### Observation Design

Target observation:

- one local observation per agent
- player-centric features
- fixed-size vector observation

Paper target:

- `55` dimensions per agent

Feature groups:

- self state
- teammate state
- opponent features
- goalkeeper features
- ball features
- context features
- agent identity features

The exact schema should be frozen in writing before implementation.

### Action Design

Use a reduced discrete action set for the `2v2` line.

Current intended actions:

- `8` movement directions
- short pass
- long pass
- shot
- sprint
- dribble

Spec issue:

- this list gives `13` actions, not `14`

Before coding, decide whether the missing action is:

- `idle`
- `high_pass`
- another release or control action

Do not let this remain ambiguous in the code.

## Implementation Phases

### Phase 0: Freeze The Study Setting

Goal:

- keep the study target fixed while the training stack stabilizes

Frozen decisions for now:

- `2` attacking RL agents
- built-in goalkeeper
- built-in defenders
- shared policy
- one main training environment
- academy tasks used later for probing

### Phase 1: Runnable Baseline

Goal:

- get a minimal `2v2` PPO pipeline running end to end

Deliverables:

- training entry point
- environment wrapper
- checkpoint save/load
- one smoke-tested training run
- replay or video output for inspection

Success condition:

- one training run completes and can be evaluated

Status:

- done for the first local baseline in `Two_V_Two/train_basic.py`

Important note:

- the public TiKick repo is useful as a code source, but it does not expose a public football training runner, so this phase uses a local trainer built on top of vendored TiKick components

### Phase 2: Training Backbone Cleanup

Goal:

- keep only the useful TiKick-derived backbone pieces and avoid project drift

Keep using:

- shared rollout collection
- PPO or MAPPO-style update logic
- football env wrapper ideas
- replay and evaluation helpers where useful

Ignore for now:

- pretrained models
- full-game setup
- large scenario-specific upstream code
- TiZero as a main training stack

### Phase 3: Fixed Observation and Action Interfaces

Goal:

- replace generic interfaces with the exact `2v2` spec

Deliverables:

- `55`-d local observation builder
- reduced action mapper
- action validation for ball-only skills
- logging that confirms mapped actions are behaving correctly

Success condition:

- both agents train from the intended local vector observation and act through the reduced action space

### Phase 4: Plain PPO Baseline

Goal:

- establish the baseline before adding regularization

Deliverables:

- stable PPO training loop
- shared policy for both agents
- basic metrics:
  - return
  - goal success rate
  - pass frequency
  - shot frequency

Success condition:

- the baseline learns something measurable beyond random behavior

### Phase 5: Evaluation Layer

Goal:

- measure progression and coordination with fixed probes

Deliverables:

- academy-task probe scripts
- goal rate logging
- return curve logging
- replay dump support
- action frequency statistics

Success condition:

- checkpoints can be compared through one repeatable evaluation suite

### Phase 6: Controlled Experiments

Goal:

- run a small, interpretable first experiment matrix

First three runs:

- plain PPO baseline
- PPO + KL prior
- PPO + reward shaping

Rules:

- do not add more variants before these three are stable
- keep the same evaluation suite across all three

### Phase 7: Human-Guided Refinement

Goal:

- use replay inspection to add targeted changes one at a time

Rules:

- inspect failures through replays
- add one intervention at a time
- rerun the same comparisons after each change

### Phase 8: Behavior Emergence Figures

Goal:

- produce the plots and tables needed for the final writeup

Metrics:

- episode return
- goal success rate
- pass success rate
- pass-to-shot conversion
- shot frequency
- action distribution over time
- coordinated attack success rate

Outputs:

- checkpoint-by-checkpoint evaluation table
- behavior curves over training
- representative replay clips

Success condition:

- we can point to when passing, positioning, and finishing emerge

## Engineering Rules

- keep PPO as the fixed optimization backbone at first
- avoid changing observation design, reward shaping, and regularization all in the same experiment
- keep one experiment change per stage whenever possible
- write down every change before large training runs
- prefer simple, inspectable code over premature optimization

## Immediate Next Steps

1. Keep the baseline scenario fixed at `academy_run_pass_and_shoot_with_keeper` while the local stack hardens.
2. Freeze the exact reduced action list.
3. Write the exact `55`-d observation schema.
4. Add evaluation probes and replay-friendly logging to the current baseline.
5. Run the first stable plain PPO baseline experiment.

## Folder Use

This folder should hold the shared planning and experiment notes for the new `2v2` line.

Recommended next documents:

- `EXPERIMENT_LOG.md`
- `OBSERVATION_SPEC.md`
- `ACTION_SPEC.md`
- `EVAL_PLAN.md`
