# Team Collaboration

## Shared Rule

Everyone runs their own experiments locally, but code changes should stay coordinated through the repo.

## Local Run Directories

Use local run directories under:

- `Two_V_Two/runs/yuhan/`
- `Two_V_Two/runs/jiang/`
- `Two_V_Two/runs/yao/`

These outputs should stay untracked.

## Suggested Ownership

- environment and action specification: one owner
- evaluation, probes, and logging: one owner
- algorithm variants such as KL or reward shaping: one owner

## Shared Workflow

1. Pull the latest `main`.
2. Make code changes on a personal branch.
3. Run experiments only inside your own local run directory.
4. Commit only code, configs, and notes that should be shared.
5. Record meaningful runs in `Two_V_Two/EXPERIMENT_LOG.md`.
