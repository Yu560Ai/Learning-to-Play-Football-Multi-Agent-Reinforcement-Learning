# Y_Fu Training Reading Guide

This file is the single place to review what has been done in the `Y_Fu/` training line so far.

## Read First

1. [MARL_ROADMAP_SPEC.md](/home/yuhan/Codes/RL/Learning-to-Play-Football-Multi-Agent-Reinforcement-Learning/Y_Fu/MARL_ROADMAP_SPEC.md)

Use this to understand the intended curriculum:

- `academy_pass_and_shoot_with_keeper`
- `academy_3_vs_1_with_keeper`
- `five_vs_five`

It explains:

- why the stages are ordered this way
- what behavior each stage is supposed to learn
- when a stage is good enough to transfer forward

Also read:

- [ACADEMY_TO_5V5_BOOTSTRAP_PLAN.md](/home/yuhan/Codes/RL/Learning-to-Play-Football-Multi-Agent-Reinforcement-Learning/Y_Fu/ACADEMY_TO_5V5_BOOTSTRAP_PLAN.md)
- [PPO_OFFLINE_RL_INTEGRATION_PLAN.md](/home/yuhan/Codes/RL/Learning-to-Play-Football-Multi-Agent-Reinforcement-Learning/Y_Fu/PPO_OFFLINE_RL_INTEGRATION_PLAN.md)
- [TRAINING_FAILURE_ARCHIVE.md](/home/yuhan/Codes/RL/Learning-to-Play-Football-Multi-Agent-Reinforcement-Learning/Y_Fu/TRAINING_FAILURE_ARCHIVE.md)

2. [multiagent_eval_report.md](/home/yuhan/Codes/RL/Learning-to-Play-Football-Multi-Agent-Reinforcement-Learning/Y_Fu/reports/multiagent_eval_report.md)

Use this to understand what the saved checkpoints actually achieved.

It summarizes outcome metrics for:

- `2_agents`
- `3_agents`
- `5_agents`
- `11_agents`

Read this before assuming a later checkpoint is better.

## Checkpoint Folders

Open these folders to see the actual saved model history:

- [academy_pass_and_shoot_with_keeper](/home/yuhan/Codes/RL/Learning-to-Play-Football-Multi-Agent-Reinforcement-Learning/Y_Fu/checkpoints/academy_pass_and_shoot_with_keeper)
- [academy_3_vs_1_with_keeper](/home/yuhan/Codes/RL/Learning-to-Play-Football-Multi-Agent-Reinforcement-Learning/Y_Fu/checkpoints/academy_3_vs_1_with_keeper)
- [five_vs_five](/home/yuhan/Codes/RL/Learning-to-Play-Football-Multi-Agent-Reinforcement-Learning/Y_Fu/checkpoints/five_vs_five)

What to look for:

- `latest.pt`
- earlier `update_*.pt` files
- whether the folder is fully populated or still in progress

Important rule:

- `latest.pt` is only the most recent checkpoint
- it is not automatically the best checkpoint

## Videos

Watch representative behavior here:

- [multiagent](/home/yuhan/Codes/RL/Learning-to-Play-Football-Multi-Agent-Reinforcement-Learning/Y_Fu/videos/multiagent)

Subfolders:

- [2_agents](/home/yuhan/Codes/RL/Learning-to-Play-Football-Multi-Agent-Reinforcement-Learning/Y_Fu/videos/multiagent/2_agents)
- [3_agents](/home/yuhan/Codes/RL/Learning-to-Play-Football-Multi-Agent-Reinforcement-Learning/Y_Fu/videos/multiagent/3_agents)
- [5_agents](/home/yuhan/Codes/RL/Learning-to-Play-Football-Multi-Agent-Reinforcement-Learning/Y_Fu/videos/multiagent/5_agents)
- [11_agents](/home/yuhan/Codes/RL/Learning-to-Play-Football-Multi-Agent-Reinforcement-Learning/Y_Fu/videos/multiagent/11_agents)

Use the videos to check:

- whether passing is purposeful
- whether teammates are actually used
- whether the team keeps shape or collapses into chaos
- whether a checkpoint that looks good in metrics also looks good in play

## Command Reference

If you want to see the exact commands used in this project, read:

- [TERMINAL_COMMANDS.md](/home/yuhan/Codes/RL/Learning-to-Play-Football-Multi-Agent-Reinforcement-Learning/Y_Fu/TERMINAL_COMMANDS.md)
- [OFFLINE_RL_COMMANDS.md](/home/yuhan/Codes/RL/Learning-to-Play-Football-Multi-Agent-Reinforcement-Learning/Y_Fu/OFFLINE_RL_COMMANDS.md)

## Short Recommended Reading Order

If you want the shortest useful path, read in this order:

1. [MARL_ROADMAP_SPEC.md](/home/yuhan/Codes/RL/Learning-to-Play-Football-Multi-Agent-Reinforcement-Learning/Y_Fu/MARL_ROADMAP_SPEC.md)
2. [multiagent_eval_report.md](/home/yuhan/Codes/RL/Learning-to-Play-Football-Multi-Agent-Reinforcement-Learning/Y_Fu/reports/multiagent_eval_report.md)
3. the checkpoint folders under [checkpoints](/home/yuhan/Codes/RL/Learning-to-Play-Football-Multi-Agent-Reinforcement-Learning/Y_Fu/checkpoints)
4. the videos under [multiagent](/home/yuhan/Codes/RL/Learning-to-Play-Football-Multi-Agent-Reinforcement-Learning/Y_Fu/videos/multiagent)
5. [PPO_OFFLINE_RL_INTEGRATION_PLAN.md](/home/yuhan/Codes/RL/Learning-to-Play-Football-Multi-Agent-Reinforcement-Learning/Y_Fu/PPO_OFFLINE_RL_INTEGRATION_PLAN.md)
6. [PPO_IQL_EXECUTION_CHECKLIST.md](/home/yuhan/Codes/RL/Learning-to-Play-Football-Multi-Agent-Reinforcement-Learning/Y_Fu/PPO_IQL_EXECUTION_CHECKLIST.md)
7. [TRAINING_FAILURE_ARCHIVE.md](/home/yuhan/Codes/RL/Learning-to-Play-Football-Multi-Agent-Reinforcement-Learning/Y_Fu/TRAINING_FAILURE_ARCHIVE.md)

## Current Interpretation

The current training story is:

- the roadmap is `2 -> 3 -> 5`
- `2_agents` did not show strong outcome improvement
- `3_agents` showed a small positive signal
- `five_vs_five` is the main target now
- Arena evaluation against Google shows the current shared `5v5` model is still below the built-in baseline

That is why the current priority is:

- use Academy to bootstrap primitives
- transfer them into `five_vs_five`
- then use offline RL only on `five_vs_five`
