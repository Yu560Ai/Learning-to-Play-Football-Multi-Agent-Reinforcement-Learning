# Y_Fu Spec

## Purpose

This file is the unified current spec for the `Y_Fu/` training line.

Use it as the short answer to:

- what the project is trying to do now
- what the main method is
- what the recommended training order is
- which documents matter most

Historical failures and older exploratory cases have been moved out of this file.

For those, read:

- [TRAINING_FAILURE_ARCHIVE.md](/home/yuhan/Codes/RL/Learning-to-Play-Football-Multi-Agent-Reinforcement-Learning/Y_Fu/TRAINING_FAILURE_ARCHIVE.md)

## Current Objective

The current final target is:

- `5_vs_5`

The main research direction is:

- multi-agent football behavior under shared-policy PPO
- curriculum transfer from Academy to `5_vs_5`
- offline RL as a `5_vs_5` improvement stage, not a replacement for curriculum

## Current Method Stack

### Online RL

- shared-policy PPO
- vectorized rollout collection
- custom reward shaping
- checkpoint selection by evaluation plus video

### Curriculum

Use Academy as a primitive-learning bootstrap:

1. `academy_run_to_score_with_keeper`
2. `academy_pass_and_shoot_with_keeper`
3. `academy_3_vs_1_with_keeper`
4. `five_vs_five`

### Offline RL

Use offline RL only after a meaningful `5_vs_5` PPO policy exists.

Recommended role:

- collect `5_vs_5` data from PPO
- train IQL on `5_vs_5`
- if IQL is better, either:
  - collect more `5_vs_5` data from IQL, or
  - initialize PPO from IQL and continue online

Important:

- do not mix Academy and `5_vs_5` data in the same initial IQL run

## Current Design Rules

### 1. `five_vs_five` is the main target

Academy is useful, but it is not the endpoint.

### 2. Academy is for primitives

Use Academy to teach:

- passing
- support movement
- simple shot creation

Do not spend most total compute there once the primitive is real.

Operational rule:

- budget Academy mainly in `env steps`
- keep Stage 1 short
- let Stage 2 absorb most Academy compute
- use Stage 3 as a transfer filter, not as an endless sink

### 3. `5_vs_5` is for transfer and realism

Use `5_vs_5` to learn:

- spacing
- transition behavior
- recovery after losing the ball
- more robust attack decisions

### 4. Offline RL is for replay-based improvement

Use offline RL to exploit `5_vs_5` trajectories more efficiently once PPO has already found some useful behavior.

### 5. `latest.pt` is not automatically best

Always choose checkpoints by:

1. evaluation
2. video behavior
3. transfer quality

not only by recency.

## Recommended Workflow

### Phase A: Academy PPO

Train and evaluate Academy stages until passing and finishing behavior is visibly present.

### Phase B: `5_vs_5` PPO

Initialize from the best Academy checkpoint and first run an early transfer check in `5_vs_5`.

Only after passing that early transfer check should the policy receive the main long `5_vs_5` budget or feed offline RL.

### Phase C: `5_vs_5` Offline RL

Collect `5_vs_5` datasets from:

- best PPO checkpoint
- exploratory PPO sampling
- weaker PPO checkpoint
- optional random baseline

Train IQL only on manifest-compatible `5_vs_5` data.

### Phase D: PPO From IQL

If IQL looks better than PPO in `5_vs_5`, initialize PPO from the best IQL checkpoint and continue online.

## Evaluation Rules

Use these principles consistently:

- deterministic evaluation is useful for reproducibility, not enough for robustness
- use multiple seeds before making strong conclusions
- use video as well as scalar metrics

Important metrics:

- `win_rate`
- `avg_goal_diff`
- `avg_score_reward`
- pass quality
- shot creation

## Current File Map

### Most important current docs

- [README.md](/home/yuhan/Codes/RL/Learning-to-Play-Football-Multi-Agent-Reinforcement-Learning/Y_Fu/README.md)
- [TRAINING_READING_GUIDE.md](/home/yuhan/Codes/RL/Learning-to-Play-Football-Multi-Agent-Reinforcement-Learning/Y_Fu/TRAINING_READING_GUIDE.md)
- [TRAINING_STAGE_LOG.md](/home/yuhan/Codes/RL/Learning-to-Play-Football-Multi-Agent-Reinforcement-Learning/Y_Fu/TRAINING_STAGE_LOG.md)
- [ACADEMY_TO_5V5_BOOTSTRAP_PLAN.md](/home/yuhan/Codes/RL/Learning-to-Play-Football-Multi-Agent-Reinforcement-Learning/Y_Fu/ACADEMY_TO_5V5_BOOTSTRAP_PLAN.md)
- [PPO_OFFLINE_RL_INTEGRATION_PLAN.md](/home/yuhan/Codes/RL/Learning-to-Play-Football-Multi-Agent-Reinforcement-Learning/Y_Fu/PPO_OFFLINE_RL_INTEGRATION_PLAN.md)
- [OFFLINE_RL_COMMANDS.md](/home/yuhan/Codes/RL/Learning-to-Play-Football-Multi-Agent-Reinforcement-Learning/Y_Fu/OFFLINE_RL_COMMANDS.md)
- [PPO_IQL_EXECUTION_CHECKLIST.md](/home/yuhan/Codes/RL/Learning-to-Play-Football-Multi-Agent-Reinforcement-Learning/Y_Fu/PPO_IQL_EXECUTION_CHECKLIST.md)

### Failure history

- [TRAINING_FAILURE_ARCHIVE.md](/home/yuhan/Codes/RL/Learning-to-Play-Football-Multi-Agent-Reinforcement-Learning/Y_Fu/TRAINING_FAILURE_ARCHIVE.md)
- [FIVE_V_FIVE_FAILURE_SOLUTION_SWEEP.md](/home/yuhan/Codes/RL/Learning-to-Play-Football-Multi-Agent-Reinforcement-Learning/Y_Fu/FIVE_V_FIVE_FAILURE_SOLUTION_SWEEP.md)

## Practical Bottom Line

The current answer for this repo is:

- keep Academy
- keep PPO as the main online method
- use offline RL on `5_vs_5`, not as a mixed-task shortcut
- move historical failure examples out of the main spec

That is the stable working direction.
