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

## Upgrade Roadmap

The project has been upgraded in this order:

1. Get the shared `football-master/football-env` working and verify basic `gfootball` import and environment creation.
2. Add a stable PPO training path in `Y_Fu` with checkpoint saving and evaluation.
3. Add a curriculum instead of jumping straight to full football:
   - `academy_run_to_score_with_keeper`
   - `academy_pass_and_shoot_with_keeper`
   - `academy_3_vs_1_with_keeper`
   - `five_vs_five`
4. Add a model zoo and preset system so different experiments can share one trainer.
5. Improve logging:
   - `episodes_finished`
   - `score_reward`
   - `episode_length_range`
   - `success_rate`
   - `goals_for`
   - `goals_against`
   - `score_examples`
6. Add seeded evaluation and video export in `Y_Fu/evaluate.py`.
7. Realize that `latest.pt` is not always the best checkpoint and start selecting earlier checkpoints by evaluation.
8. Find that `academy_3_vs_1_with_keeper/update_90.pt` was a better attacking checkpoint than `latest.pt`.
9. Transfer from `academy_3_vs_1_with_keeper/update_90.pt` into `five_vs_five`.
10. Increase `five_vs_five` rollout length from `512` to `1024` for longer match context per PPO update.
11. Add offense-focused reward shaping:
   - pass success reward
   - pass failure penalty
   - shot attempt reward
   - attacking possession reward
12. Add a second shaping pass:
   - pass progress reward
   - final-third entry reward
   - possession retention reward
   - own-half turnover penalty

Current intended next training command after the last upgrade:

```bash
python -u Y_Fu/train.py --preset five_vs_five --device cpu --rollout-steps 1024 --total-timesteps 1000000 --init-checkpoint Y_Fu/checkpoints/five_vs_five/update_140.pt --pass-success-reward 0.08 --pass-failure-penalty 0.06 --pass-progress-reward-scale 0.08 --shot-attempt-reward 0.03 --attacking-possession-reward 0.0015 --final-third-entry-reward 0.04 --possession-retention-reward 0.0008 --own-half-turnover-penalty 0.04
```

## Current Milestone

Latest evaluated `five_vs_five` result before the new shaping rerun:

- trained policy:
  - `avg_goal_diff = -1.900`
  - `avg_score_reward = -1.900`
  - `win_rate = 0.000`
- random policy:
  - `avg_goal_diff = -2.800`
  - `avg_score_reward = -2.800`
  - `win_rate = 0.000`
- delta vs random:
  - `avg_goal_diff = +0.900`

Interpretation:

- this is better than the older `five_vs_five` runs
- the policy is clearly better than random
- but it still cannot score reliably and still loses most matches
- the main remaining bottleneck is offense, not basic survival

## What A Checkpoint Is

A checkpoint is a saved snapshot of training state at a particular update. In this project, a checkpoint stores:

- model weights
- optimizer state
- training config
- observation/action dimensions
- update number
- total agent steps

So a checkpoint lets you:

- evaluate a past version of the policy
- continue training from that saved state
- compare earlier and later policies

`latest.pt` is only the most recently saved checkpoint in that preset folder. It is not guaranteed to be the best-performing checkpoint. Earlier checkpoints such as `update_90.pt` can be better if the policy later drifts into a weaker or more conservative behavior.

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

`episodes_finished` means how many episodes reached a terminal state during the current fixed rollout window. PPO collects a fixed chunk of steps per update, so an episode can continue across update boundaries and finish in a later update.

For academy scenarios such as `academy_3_vs_1_with_keeper`, an episode usually ends when:

- a goal is scored
- the defending side gains possession
- the ball goes out of play

This means:

- short episode + positive `score_reward` usually means quick success
- short episode + zero `score_reward` usually means turnover or ball out
- positive `episode_return` with zero `score_reward` means shaping reward was collected, but no actual goal was scored

For `five_vs_five`, episodes behave more like full matches. In practice, an episode usually ends at the scenario's full game terminal condition, often the built-in match length limit. That is why `episode_length` can be around `3001` and why many updates can show `episodes_finished=0`.

This means:

- academy tasks have short, frequent terminal events
- `five_vs_five` has long episodes and much sparser completed-episode feedback

`rollout_steps` is the fixed number of environment steps collected per PPO update. If the rollout window is short relative to episode length, that is not an error by itself. It simply means:

- many updates will contain no completed episodes
- training logs will often show `episode_return=n/a`
- reward and value learning become noisier because terminal feedback is seen less often

It only becomes a practical problem when the rollout is so short that learning is too slow or unstable for the long-horizon task.

Example for `five_vs_five`:

- one full match can last about `3001` steps
- one PPO update with `rollout_steps=512` collects `512` environment steps

So one full match spans about `3001 / 512 ~= 5.86` rollout windows, which is roughly 6 updates. That is why:

- many consecutive updates can show `episodes_finished=0`
- then one later update shows `episodes_finished=1`

This is normal. The episode is continuing across several PPO updates and only finishes when one update happens to include the terminal step.

## Value Function In Sparse Reward

In the current setup, the value function is learned from rollouts using:

- rewards collected along the episode
- bootstrapping with `V(next_state)`
- GAE smoothing

So even if the final goal happens late, the critic can still learn earlier estimates such as:

- this state tends to lead to success more often
- this state usually collapses into failure

That is the main purpose of the value function in sparse-reward RL.

There are still clear limits:

- if reward is very rare
- if episodes are long
- if the state/action space is large

then the value estimate becomes noisy and hard to learn. This is one of the main reasons full football is much harder than the academy curriculum stages.

MCTS is usually not the practical next step for this project. Football is harder for MCTS because:

- it is real-time and long-horizon
- it is multi-agent
- action branching compounds over time
- control feels continuous even though the action space is discrete
- online tree search would be expensive to run repeatedly

More practical ways to handle sparse reward here are:

- curriculum learning
- reward shaping
- better checkpoint selection
- larger training budget
- role-aware architectures
- centralized critic or MAPPO-style methods

## Pure RL Vs Guided Shaping

The reward shaping used in `Y_Fu` injects human knowledge. This is less pure than an AlphaZero-style approach and should be treated as a practical engineering choice, not as a claim of fully autonomous discovery.

Tradeoff:

- minimal human input is more elegant and more faithful to pure RL
- but it is much harder to train in this football setting
- guided shaping is less pure
- but more practical for a small project with limited compute

In practice for this project, some shaping is often necessary just to get traction. Without it, PPO may never discover useful attacking behavior at all.

## Important Checkpoint Note

- New presets save into separate folders so they do not overwrite each other.
- `Y_Fu/checkpoints/latest.pt` was restored to the older longer run and should not be treated as the best model automatically.
- Prefer evaluating the checkpoint inside the preset-specific folder you just trained.
