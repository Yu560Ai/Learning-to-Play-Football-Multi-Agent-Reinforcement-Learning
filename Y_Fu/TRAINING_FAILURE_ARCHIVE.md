# Training Failure Archive

## Purpose

This file stores important past failure cases and the lesson from each one.

It is not the main current spec.

Use it when you want to remember:

- what already failed
- why it failed
- what rule changed afterward

## Case 1: Shaped Return Improved But The Task Was Not Solved

Observed in early `academy_pass_and_shoot_with_keeper` work:

- PPO improved shaped return
- but the policy still did not reliably score
- stage completion was not the same as stage success

Main lesson:

- never treat shaped return alone as proof that the football behavior is good
- always check actual scoring and video

Where to verify:

- [TRAINING_STAGE_LOG.md](/home/yuhan/Codes/RL/Learning-to-Play-Football-Multi-Agent-Reinforcement-Learning/Y_Fu/TRAINING_STAGE_LOG.md)

## Case 2: `latest.pt` Was Not The Best Checkpoint

Observed repeatedly in curriculum stages:

- a later checkpoint was sometimes worse than an earlier one
- transfer quality did not always correlate with recency

Main lesson:

- checkpoint choice must be based on evaluation plus video
- `latest.pt` is only the most recent checkpoint, not the best one

Practical rule:

- evaluate several nearby checkpoints before transfer

## Case 3: `academy_*` Run Finished But The Stage Did Not Pass

Observed in current logged curriculum status:

- `academy_pass_and_shoot_with_keeper` runs completed
- but the stage still did not pass its target threshold
- `academy_3_vs_1_with_keeper` was stopped early and also did not pass

Main lesson:

- "run completed" and "stage passed" must stay separate
- curriculum progression needs a gate, not just a time budget

Where to verify:

- [TRAINING_STAGE_LOG.md](/home/yuhan/Codes/RL/Learning-to-Play-Football-Multi-Agent-Reinforcement-Learning/Y_Fu/TRAINING_STAGE_LOG.md)

## Case 4: Pure `5_vs_5` PPO Looked Like It Did Not Know Basic Passing

Observed failure pattern:

- after a long `5_vs_5` PPO run, the policy still looked primitive
- players did not show clean passing or organized attack behavior
- the game often looked like motion without real football structure

Main lesson:

- `5_vs_5` from scratch is too hard as the first place to learn primitives in the current setup
- Academy is useful as a bootstrap stage

Where to verify:

- [ACADEMY_TO_5V5_BOOTSTRAP_PLAN.md](/home/yuhan/Codes/RL/Learning-to-Play-Football-Multi-Agent-Reinforcement-Learning/Y_Fu/ACADEMY_TO_5V5_BOOTSTRAP_PLAN.md)
- [FIVE_V_FIVE_FAILURE_SOLUTION_SWEEP.md](/home/yuhan/Codes/RL/Learning-to-Play-Football-Multi-Agent-Reinforcement-Learning/Y_Fu/FIVE_V_FIVE_FAILURE_SOLUTION_SWEEP.md)

## Case 5: Dense Rewards Could Encourage Empty Play

Observed diagnosis:

- a policy could move forward or maintain harmless possession
- without actually creating shots or dangerous attacks

Main lesson:

- reward shaping must emphasize real attack creation
- not only generic forward progress or safe possession

Follow-up rule:

- compare reward variants systematically
- inspect goals, shots, and videos, not only reward curves

Where to verify:

- [FIVE_V_FIVE_FAILURE_SOLUTION_SWEEP.md](/home/yuhan/Codes/RL/Learning-to-Play-Football-Multi-Agent-Reinforcement-Learning/Y_Fu/FIVE_V_FIVE_FAILURE_SOLUTION_SWEEP.md)

## Case 6: Fixed-Seed Deterministic Evaluation Was Useful But Easy To Overtrust

Observed issue:

- deterministic evaluation with one seed is good for reproducibility
- but it can replay very similar episodes and overstate confidence

Main lesson:

- use fixed-seed evaluation for comparison
- use multiple seeds before making stronger conclusions

Where to verify:

- [TRAINING_STAGE_LOG.md](/home/yuhan/Codes/RL/Learning-to-Play-Football-Multi-Agent-Reinforcement-Learning/Y_Fu/TRAINING_STAGE_LOG.md)

## Case 7: Single-Player Side Paths Did Not Match The Main Research Goal

Observed issue:

- single-player lines were useful as engineering references
- but they did not match the main multi-agent paper direction

Main lesson:

- keep the main effort on the shared multi-agent line
- treat single-player baselines as supplemental, not central

## Case 8: Academy And `5_vs_5` Should Not Be Naively Mixed Offline

Current structural lesson:

- Academy and `5_vs_5` have different task distributions
- they are both useful, but not as one untagged offline dataset

Main lesson:

- Academy should feed PPO transfer
- offline RL should start from `5_vs_5` datasets

Where to verify:

- [PPO_OFFLINE_RL_INTEGRATION_PLAN.md](/home/yuhan/Codes/RL/Learning-to-Play-Football-Multi-Agent-Reinforcement-Learning/Y_Fu/PPO_OFFLINE_RL_INTEGRATION_PLAN.md)

## Bottom Line

The historical failures point to a stable current rule set:

- use Academy for primitive learning
- use `5_vs_5` for the real target
- gate stage transfer by behavior, not only by runtime
- choose checkpoints by evidence, not recency
- use offline RL on `5_vs_5`, not as a mixed-task shortcut
