# SaltyFish Upgrade Spec

## Purpose

This file records the structural gap between the local `Y_Fu/saltyfish_baseline/` code and the public competition-style SaltyFish descriptions, plus the scoped upgrades implemented in this repository.

Use this file before continuing SaltyFish-related work.

Important scope note:

- this baseline is a **single-player** control setup inside a full `11_vs_11_kaggle` match
- it does **not** train a full learned team
- it is therefore not the main branch for studying multi-agent cooperation

## What The Public Solution Style Had

Public summaries of the Google Football competition solutions consistently emphasize:

- richer training loops than plain local PPO
- richer reward shaping than sparse goal-only reward
- recurrent memory
- richer state features from raw football observations
- opponent diversity, self-play, and league-style training
- in some teams, imitation / replay learning

Relevant references used for this summary:

- Kaggle writeup page: `SaltyFish: Current 2nd place solution`
- Tencent / WeKick summary
- Google Research Football observation docs

## What The Local Baseline Originally Had

Before this upgrade, the local SaltyFish-inspired baseline was intentionally much smaller:

- single-machine PPO
- single-player control only
- `simple115v2` only
- grouped feed-forward MLP
- sparse `scoring` reward only
- reduced action set
- no replay learning
- no opponent pool or league training
- no recurrent state

This was already stated in [SPEC.md](/home/yuhan/Codes/RL/Learning-to-Play-Football-Multi-Agent-Reinforcement-Learning/Y_Fu/SPEC.md).

## Main Gap Diagnosis

The poor result was not surprising. The local code had copied only the easiest visible parts of the public description:

- competition scenario selection
- `simple115v2`
- a grouped feature encoder

But the likely high-leverage pieces were still missing:

1. Sparse reward remained mostly unsolved.
2. The agent did not see sticky actions, fatigue, cards, or offside-aware teammate signals.
3. The model had no memory.
4. Training had no opponent diversity or replay supervision.
5. The original default rollout was too short relative to a full match.

Also, even after these upgrades, this remains a single-player setup:

- one learned controlled player
- built-in teammates around that player
- full `11v11` match context, but not full-team learned coordination

So this branch is useful as a competition-style baseline, but it is not the primary research branch for multi-agent cooperation.

## GPU Status

As checked in the local `football-env` on 2026-03-30:

- `torch 2.11.0+cu130`
- `torch.cuda.is_available() == False`
- `torch.cuda.device_count() == 0`
- `nvidia-smi` failed with `GPU access blocked by the operating system`

So the RTX 4060 is not usable from the current environment until the driver / runtime stack is fixed.

## Implemented Upgrade

The repo now contains a stronger local baseline, still single-machine, but closer to the public competition style where feasible.

This should still be interpreted as:

- single-player PPO baseline
- in a full `11_vs_11_kaggle` environment
- not a learned multi-agent team

### Observation Upgrade

The SaltyFish baseline now augments `simple115v2` with engineered features from raw observations:

- active player position and direction
- active player fatigue, yellow card, and role
- active-to-ball relative state
- active-to-goal relative state
- nearest teammate and opponent relative geometry
- team and opponent centroid relative geometry
- heuristic offside flags for teammates
- sticky actions
- full team fatigue vectors
- full team yellow-card vectors
- ball ownership indicators
- active-has-ball flag
- normalized score difference
- normalized steps left

The observation size is now:

- old baseline: `115`
- upgraded baseline: `222`

### Reward Upgrade

The local baseline now adds competition-style shaping on top of `scoring`:

- possession gain reward
- possession loss penalty
- small reward for team possession
- small penalty for opponent possession
- successful pass reward
- progressive pass reward based on forward ball movement
- carry-progress reward for the same ball carrier moving forward
- attacking-third possession reward
- shots-with-ball reward
- reduced possession-loss penalty in advanced attacking areas
- explicit penalty for losing the ball out for throw-ins / goal-kicks

This is still not a full reproduction of the public competition shaping, but it directly addresses the sparse-reward failure mode.

### Training Upgrade

The training defaults were upgraded to a more appropriate local baseline:

- `total_timesteps = 2_000_000`
- `rollout_steps = 1024`
- `learning_rate = 1e-4`
- `gamma = 0.993`

### Checkpoint Compatibility Upgrade

The trainer now supports partial initialization from older `115`-dim checkpoints:

- compatible tensors are loaded
- incompatible tensors are skipped
- this allows old baseline weights to seed the upgraded model instead of failing on shape mismatch

## What Is Still Missing

This upgraded baseline is still not the full competition system.

Still missing:

- recurrent policy or LSTM memory
- real self-play league or opponent pool
- imitation / replay learning from strong agents
- local ladder-style evaluation against multiple opponents
- more principled action-set experiments

Those are the next tier of upgrades, not the first one.

## Recommended Next Commands

Train the upgraded baseline from scratch:

```bash
cd ~/Codes/RL/Learning-to-Play-Football-Multi-Agent-Reinforcement-Learning
source football-master/football-env/bin/activate
python -u Y_Fu/train_saltyfish.py --device cpu
```

Seed the upgraded baseline from the older checkpoint family:

```bash
cd ~/Codes/RL/Learning-to-Play-Football-Multi-Agent-Reinforcement-Learning
source football-master/football-env/bin/activate
python -u Y_Fu/train_saltyfish.py --device cpu --init-checkpoint Y_Fu/checkpoints/saltyfish_baseline/latest.pt
```

Evaluate the upgraded baseline:

```bash
cd ~/Codes/RL/Learning-to-Play-Football-Multi-Agent-Reinforcement-Learning
source football-master/football-env/bin/activate
python Y_Fu/evaluate_saltyfish.py --checkpoint Y_Fu/checkpoints/saltyfish_baseline/latest.pt --episodes 10 --deterministic --compare-random --device cpu
```

The evaluation summary now reports:

- `avg_goal_diff`
- `avg_goals_for`
- `avg_goals_against`
- `win_rate`
- `draw_rate`

## Decision Rule

Use this sequence:

1. Run the upgraded baseline for a meaningful training budget.
2. Compare against random on at least 10 episodes.
3. If the upgraded baseline still cannot beat random, stop investing in architecture-only changes and move to one of:
   - recurrent PPO
   - opponent-pool / self-play training
   - replay or imitation learning
