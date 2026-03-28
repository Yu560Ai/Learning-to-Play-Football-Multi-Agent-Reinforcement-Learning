# Y_Fu Football RL Baseline

This folder contains a self-contained **PyTorch PPO baseline** for the local `football-master` environment in this repository.

The implementation is designed as a practical starting point for your project:

- it imports the local `football-master` source tree automatically
- it trains a **shared policy** for all controlled left-team players
- it works well with `simple115v2` observations for normal `11_vs_11_*` scenarios
- it keeps all custom code inside `Y_Fu` without modifying the upstream environment

## What Is Implemented

- `train.py`: trains a PPO agent
- `train_lightning.py`: launches a very fast starter run
- `evaluate.py`: loads a saved checkpoint and runs evaluation episodes
- `yfu_football/envs.py`: environment wrapper for Google Research Football
- `yfu_football/model.py`: actor-critic network
- `yfu_football/ppo.py`: rollout collection, GAE, PPO updates, checkpointing

## Default Training Setup

The default command trains on:

- scenario: `11_vs_11_easy_stochastic`
- observation: `simple115v2`
- reward: `scoring,checkpoints`
- controlled players: `11`

That means this baseline is already aligned with the multi-agent direction of the project, but it uses **parameter-sharing PPO** rather than a centralized-critic MAPPO implementation.

## Before Running

You still need a working Google Research Football build. If the local source import works but the engine is not compiled yet, build/install the environment from the `football-master` folder first.

You also need PyTorch available in your Python environment.

In the current workspace verification, the missing module is:

```text
gfootball_engine
```

So the next runtime setup step is to build the football engine successfully before launching training.

## Train

From the repository root:

```bash
python Y_Fu/train.py
```

For the fastest first run, use the lightning preset:

```bash
python Y_Fu/train_lightning.py
```

This quick-start preset uses:

- scenario: `academy_empty_goal_close`
- observation: `extracted`
- controlled players: `1`
- timesteps: `20000`
- smaller network: `128 128`

The same run is also available from the main trainer:

```bash
python Y_Fu/train.py --preset lightning
```

Example with a smaller custom quick test run:

```bash
python Y_Fu/train.py --total-timesteps 20000 --rollout-steps 128 --save-interval 2
```

Example keeping the football match setting but shrinking it:

```bash
python Y_Fu/train.py --preset small_11v11
```

## Evaluate

```bash
python Y_Fu/evaluate.py --checkpoint Y_Fu/checkpoints/latest.pt --episodes 3 --deterministic
```

Compare your checkpoint against a random-action benchmark:

```bash
python Y_Fu/evaluate.py --checkpoint Y_Fu/checkpoints/latest.pt --episodes 5 --deterministic --compare-random
```

Benchmark metrics reported:

- `avg_return`
- `avg_score_reward`
- `avg_goal_diff`
- `win_rate`
- `avg_length`

## Notes

- `simple115v2` is intended for normal-game scenarios, especially `11_vs_11_*`.
- Episode return in the logs is the **mean reward across controlled players** over the episode.
- `score_reward`, `goal_diff`, and `win_rate` are useful to compare your policy against the random baseline.
- If you later want true MAPPO, this code is a good baseline to extend by replacing the value function with a centralized critic.
