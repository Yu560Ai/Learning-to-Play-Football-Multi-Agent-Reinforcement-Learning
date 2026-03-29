# Y_Fu Spec

## Purpose

This file is the working spec for the `Y_Fu/` training code.

If a future session continues this work, start from this file first.

## Environment

- Repository root:
  - `~/Codes/RL/Learning-to-Play-Football-Multi-Agent-Reinforcement-Learning`
- Shared environment:
  - `football-master/football-env`
- Activate from repository root:

```bash
source football-master/football-env/bin/activate
```

## Main Entry Points

- Training:
  - `python Y_Fu/train.py ...`
- Lightning sanity check:
  - `python Y_Fu/train_lightning.py --device cpu`
- Evaluation:
  - `python Y_Fu/evaluate.py ...`

## Current Design

- Keep one shared trainer implementation in `Y_Fu/yfu_football/ppo.py`
- Keep one evaluation script in `Y_Fu/evaluate.py`
- Switch experiments by preset, not by creating many separate trainer files

## Available Model Types

- `cnn`
- `residual_mlp`
- `separate_mlp`
- `mlp`
- `auto`

## Recommended Curriculum

Run these in order:

1. `academy_run_to_score_with_keeper`
2. `academy_pass_and_shoot_with_keeper`
3. `academy_3_vs_1_with_keeper`
4. `five_vs_five`
5. `small_11v11`
6. `full_11v11_residual`

## Current Presets

- `lightning`
- `academy_run_to_score_with_keeper`
- `academy_pass_and_shoot_with_keeper`
- `academy_3_vs_1_with_keeper`
- `five_vs_five`
- `small_11v11`
- `small_11v11_wide`
- `full_11v11_residual`
- `full_11v11_wide`

## Recommended Commands

### Train

```bash
python Y_Fu/train.py --preset academy_run_to_score_with_keeper --device cpu
python Y_Fu/train.py --preset academy_pass_and_shoot_with_keeper --device cpu
python Y_Fu/train.py --preset academy_3_vs_1_with_keeper --device cpu
python Y_Fu/train.py --preset five_vs_five --device cpu
python Y_Fu/train.py --preset small_11v11 --device cpu
python Y_Fu/train.py --preset full_11v11_residual --device cpu
```

### Evaluate

```bash
python Y_Fu/evaluate.py --checkpoint Y_Fu/checkpoints/latest.pt --episodes 3 --deterministic --device cpu
python Y_Fu/evaluate.py --checkpoint Y_Fu/checkpoints/latest.pt --episodes 5 --deterministic --compare-random --device cpu
python Y_Fu/evaluate.py --checkpoint Y_Fu/checkpoints/latest.pt --episodes 1 --deterministic --render --device cpu
python Y_Fu/evaluate.py --checkpoint Y_Fu/checkpoints/latest.pt --episodes 1 --deterministic --device cpu --save-video --video-dir Y_Fu/videos
```

For preset-specific checkpoints, replace `Y_Fu/checkpoints/latest.pt` with the corresponding path, for example:

```bash
Y_Fu/checkpoints/full_11v11_residual/latest.pt
Y_Fu/checkpoints/five_vs_five/latest.pt
```

## Current Diagnosis

- The old full `11_vs_11` baseline is structurally weak.
- The agent showed bad behavior in video:
  - goalkeeper leaves position
  - ball carrier dribbles badly instead of passing
- Main reasons:
  - one shared policy for all roles
  - no role-specific reward
  - checkpoint reward biases forward dribbling
  - no centralized critic
  - full `11_vs_11` is too hard as a starting point

## Logging

The PPO trainer now prints:

- `episodes_finished`
- `episode_return`
- `score_reward`
- `episode_length`
- `episode_length_range`
- `success_rate`

`success_rate` is computed per completed episode as:

- win if final score is available and left score > right score
- otherwise success if cumulative `score_reward > 0`

## Important Checkpoint Note

- New presets save into separate folders so they do not overwrite each other.
- `Y_Fu/checkpoints/latest.pt` was restored to the older longer run and should not be treated as the best model automatically.
- Prefer evaluating the checkpoint inside the preset-specific folder you just trained.
