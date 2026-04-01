# Learning Football Strategies with Multi-Agent Reinforcement Learning

## Work Note

This repository is for our group project on Google Research Football.

## Current Project Target

The current main target of the project is:

- `5_vs_5` multi-agent football

In the current team setup, this means:

- the learned policy controls `4` left-side outfield agents
- the goalkeeper is not a learned agent
- the goalkeeper remains controlled by the environment

So the active project target is:

- `5v5` football with `4` learned outfield agents and `1` environment-controlled goalkeeper

This is the main line for training, evaluation, and reporting.

## Folder Ownership

- `X_Jiang/`: Jiang's code, experiments, notes, and checkpoints
- `Y_Fu/`: Fu's code, experiments, notes, and checkpoints
- `Y_Yao/`: Yao's code, experiments, notes, and checkpoints
- `football-master/`: local Google Research Football environment

Each member should work only inside their own folder for custom training code.

Each member can use their own GPT or coding assistant to help generate code, debug issues, and build training pipelines, but the generated code should stay in that member's own folder.

## Environment

`football-master/` is the shared football environment for this project.

Before running any Python code that uses `gfootball`, activate the environment from the repository root:

```bash
source football-master/football-env/bin/activate
```

Or:

```bash
cd football-master
source football-env/bin/activate
```

When activation succeeds, the terminal prompt should show `(football-env)`.

## Working Rule

1. Activate `(football-env)` first.
2. Go back to the repository root if needed.
3. Run or develop code inside your own folder only.
4. Do not put custom training code directly inside `football-master/`.

## Quick Evaluation Demo

Before starting your own implementation, you can watch the current example result from `Y_Fu/`.

From the repository root:

```bash
source football-master/football-env/bin/activate
python Y_Fu/evaluate.py --checkpoint Y_Fu/checkpoints/latest.pt --episodes 1 --deterministic --render --device cpu
```

This opens a rendered evaluation episode so you can confirm that the environment, checkpoint, and evaluation pipeline are working.

## Current Example

Fu's current PPO baseline is in `Y_Fu/`.
