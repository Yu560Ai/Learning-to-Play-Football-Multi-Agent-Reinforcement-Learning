# PPO IQL Execution Checklist

## Purpose

This checklist is the shortest operational version of the current hybrid plan:

- Academy PPO
- `5_vs_5` PPO
- `5_vs_5` IQL
- PPO fine-tuning from IQL

## Phase 1: Academy PPO

Before leaving Academy, confirm:

1. the policy visibly passes instead of only dribbling
2. the receiver is used in attack
3. the checkpoint is chosen by behavior and evaluation, not only by recency

If these are false:

- do not move to `5_vs_5` yet

## Phase 2: `5_vs_5` PPO

Before starting offline RL, confirm:

1. the model was initialized from the best Academy checkpoint
2. the `5_vs_5` policy shows at least some pass-and-shoot transfer
3. one best checkpoint and one weaker checkpoint are selected

If the `5_vs_5` PPO policy still looks fully primitive-free:

- fix PPO transfer or shaping first

## Phase 3: `5_vs_5` Offline Pilot

Run the pilot only on `5_vs_5` data.

Confirm:

1. best PPO dataset collected
2. exploratory PPO dataset collected
3. weaker PPO dataset collected
4. random dataset collected
5. every directory contains a valid `manifest.json`

Do not mix Academy datasets into this pilot.

## Phase 4: `5_vs_5` IQL Pilot

Confirm:

1. `train_iql.py` runs end-to-end
2. `best.pt` is saved
3. `evaluate_iql.py` runs
4. IQL policy behavior is sane in video or metrics

If any of these fail:

- stop and fix the pipeline before a full run

## Phase 5: Full `5_vs_5` IQL

Confirm:

1. full `5_vs_5` PPO dataset collection is complete
2. full IQL training is complete
3. best IQL checkpoint is evaluated against PPO

Use the same evaluation style for both:

- `20` deterministic episodes
- same scenario
- same seed family

## Phase 6: Decision Gate

If IQL is better than PPO in real behavior:

- collect more `5_vs_5` data from IQL
- or initialize PPO from IQL and continue online

If IQL is not better:

- do not continue offline iterations automatically
- inspect dataset quality and PPO behavior first

## Hard Rules

- Academy is curriculum, not the final offline dataset target.
- `5_vs_5` is the main offline RL target.
- Do not trust `latest.pt` automatically.
- Use video as well as outcome metrics.
