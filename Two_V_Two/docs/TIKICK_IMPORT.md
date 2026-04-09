# TiKick Import Notes

## Source

- Upstream repo: `https://github.com/TARTRL/TiKick`
- Imported on: `2026-04-09`
- Imported into: `Two_V_Two/third_party/tikick/`
- License copied from upstream: `Two_V_Two/third_party/tikick/LICENSE`

## What Was Copied

Only the minimal public code that is useful for the `2v2` line was vendored:

- `tmarl/`
- `scripts/football/`

These now live at:

- `Two_V_Two/third_party/tikick/tmarl/`
- `Two_V_Two/third_party/tikick/scripts/football/`

## Important Correction

The public TiKick repository does not contain an `onpolicy/` directory.
Its main backbone is under `tmarl/`.

Also, the public repo is not a full public training release for football.
It includes the MAPPO components, replay buffer, networks, evaluator path, and GRF wrapper, but it does not include a public football training runner script under `scripts/football/`.

So for this project, TiKick should be treated as a vendored code source for:

- PPO or MAPPO building blocks
- replay buffer
- policy network
- football environment wrapper
- evaluator helpers

It should not be treated as a ready-to-run end-to-end training project.

## Main Files To Reuse Next

These are the most relevant files for the `Two_V_Two` implementation:

- PPO or MAPPO loss: `Two_V_Two/third_party/tikick/tmarl/algorithms/r_mappo_distributed/mappo_algorithm.py`
- policy module: `Two_V_Two/third_party/tikick/tmarl/algorithms/r_mappo_distributed/mappo_module.py`
- policy network: `Two_V_Two/third_party/tikick/tmarl/networks/policy_network.py`
- shared rollout buffer: `Two_V_Two/third_party/tikick/tmarl/replay_buffers/normal/shared_buffer.py`
- football env wrapper: `Two_V_Two/third_party/tikick/tmarl/envs/football/football.py`
- football evaluator entry: `Two_V_Two/third_party/tikick/tmarl/runners/football/football_evaluator.py`
- football replay helpers: `Two_V_Two/third_party/tikick/scripts/football/`

## What We Intentionally Did Not Copy

The following upstream folders were left out on purpose:

- `models/`
- `results/`
- `docs/`

This keeps `Two_V_Two` small and makes later `2v2` modifications easier to control.

## Practical Next Step

Build the new `2v2` code around this vendored base instead of editing upstream-style code in place everywhere:

- create a local `Two_V_Two/env/` wrapper for the exact `2v2` scenario
- create a local training entry point in `Two_V_Two/training/`
- adapt the TiKick football wrapper to the target observation and action interfaces
- keep third-party code as close to upstream as possible unless a direct patch is necessary
