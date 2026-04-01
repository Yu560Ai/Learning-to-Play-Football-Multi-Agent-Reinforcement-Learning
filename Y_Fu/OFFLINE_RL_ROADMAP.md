# Offline RL Roadmap For `Y_Fu`

## Goal

Use offline RL to improve a real `5_vs_5` PPO policy after PPO has already learned something useful online.

This roadmap is intentionally narrow:

- target task: `5_vs_5`
- method: discrete IQL
- role: replay-based refinement, not curriculum replacement

## Current Training Line

The current project direction is:

1. Academy PPO for primitive learning
2. `5_vs_5` PPO for transfer and live adaptation
3. `5_vs_5` offline RL for replay reuse
4. PPO fine-tuning from IQL if IQL looks better

The key implication is simple:

- Academy feeds PPO transfer
- `5_vs_5` feeds offline RL
- do not start by mixing Academy and `5_vs_5` into one IQL dataset

## Why Offline RL Is Worth Trying

`5_vs_5` PPO is expensive and each trajectory is used once.

Offline RL can help by:

- reusing valuable `5_vs_5` transitions many times
- learning from mixed-quality `5_vs_5` checkpoints
- improving action selection without paying full online cost again

Offline RL is not expected to:

- invent football primitives from nothing
- replace Academy curriculum
- solve transfer by itself

## Why IQL First

Use discrete IQL for the first offline pass because:

- the action space is discrete (`19` actions)
- the current code already matches this setup
- IQL works reasonably well on sub-optimal datasets
- it is simpler than jumping directly to a more conservative method like discrete CQL

If IQL is clearly unstable or overestimates too aggressively, then consider discrete CQL later.

## Data Rule

For the first offline iteration:

- collect only `5_vs_5` data
- use the best PPO checkpoint, a noisy version of it, and one weaker earlier checkpoint
- add random-policy coverage only if state coverage is obviously too narrow

The first offline run should not mix Academy and `5_vs_5` data because the task distributions are too different.

## Reward Rule

Default to the same shaped reward semantics used by the source checkpoint when storing dataset `reward`.

That keeps offline optimization aligned with the PPO objective that actually produced the data.

Use `score_reward` as a comparison target or ablation, not as the default first run.

## Code Assumptions

The offline line reuses the existing stack:

- [collect_offline_data.py](/home/yuhan/Codes/RL/Learning-to-Play-Football-Multi-Agent-Reinforcement-Learning/Y_Fu/collect_offline_data.py)
- [train_iql.py](/home/yuhan/Codes/RL/Learning-to-Play-Football-Multi-Agent-Reinforcement-Learning/Y_Fu/train_iql.py)
- [evaluate_iql.py](/home/yuhan/Codes/RL/Learning-to-Play-Football-Multi-Agent-Reinforcement-Learning/Y_Fu/evaluate_iql.py)
- [yfu_football/offline_dataset.py](/home/yuhan/Codes/RL/Learning-to-Play-Football-Multi-Agent-Reinforcement-Learning/Y_Fu/yfu_football/offline_dataset.py)
- [yfu_football/iql.py](/home/yuhan/Codes/RL/Learning-to-Play-Football-Multi-Agent-Reinforcement-Learning/Y_Fu/yfu_football/iql.py)

Operationally, the important support is:

- collector can read PPO checkpoints
- collector can read IQL checkpoints
- dataset loader validates manifest compatibility
- PPO can be resumed from an IQL checkpoint

## Initial Defaults

Start with:

- `expectile = 0.7`
- `temperature = 3.0`
- `gamma = 0.993`
- `tau = 0.005`
- `batch_size = 4096`
- `learning_rate = 3e-4`

If training looks unstable:

- lower `temperature`
- move `expectile` toward `0.6`
- reduce `batch_size` only if memory forces it

## Evaluation Rule

Compare IQL and PPO on the same target:

- `5_vs_5`
- built-in scripted opponent
- deterministic evaluation
- same seed family

Main metrics:

- `win_rate`
- `avg_goal_diff`
- `avg_score_reward`
- video evidence of passing and shot creation

If IQL is numerically close but visibly cleaner, that still matters.

## Iteration Loop

### Iteration 0

- collect the first `5_vs_5` PPO dataset
- run pilot IQL
- if healthy, run full IQL
- compare best IQL against best PPO

### Iteration 1

If IQL is better:

- collect more `5_vs_5` data from the improved behavior policy
- merge it into the existing `5_vs_5` corpus
- train the next IQL round

### Return To PPO

If the best IQL policy is better than PPO:

- initialize PPO from the best IQL checkpoint
- continue online `5_vs_5` training

## Success Criterion

Keep offline RL only if it improves at least one real `5_vs_5` outcome:

- visible passing quality
- shot creation
- early attack structure
- `avg_goal_diff`
- sample efficiency relative to PPO alone

If it does not improve the real policy, stop iterating on it.

## Bottom Line

The compact rule for this repo is:

- Academy teaches primitives through PPO
- `5_vs_5` PPO adapts those primitives online
- `5_vs_5` offline RL reuses and improves those trajectories
- PPO can resume from the improved offline policy
