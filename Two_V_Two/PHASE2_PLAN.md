# Phase 2 Plan: Multi-Agent Structure Study

## Goal

Phase 2 studies whether agent identity and centralized critic structure can stabilize and extend the partial cooperative behavior induced by `R3`, compared against the stable non-cooperative `R2` baseline.

## Fixed Components

The following stay fixed across Phase 2:

- the same reduced 2-agent GRF attacking task
- the same local observation backbone, except for explicit identity additions where required
- the same reduced action space
- the same training budget within each architecture comparison
- reward variants limited to `r2_progress` and `r3_assist`

## Compared Architectures

### A. Shared PPO Baseline

- current shared-policy baseline
- no explicit agent identity
- no centralized critic

### B. Shared PPO + Agent Identity

- same shared actor as baseline
- append agent identity features to each agent local observation
- no centralized critic

### C. MAPPO Centralized Critic

- shared actor with agent identity
- centralized critic using joint information from both controlled agents
- actor still uses local observations

## Hypothesis

- `R2` should remain mostly non-cooperative across all structures
- `R3` may benefit from identity and centralized critic structure
- the main Phase 2 target is whether pass-level coordination becomes more stable and whether pass-to-shot or assist structure begins to appear

## Deliverables

- `PHASE2_PLAN.md`
- implementation changes for the three structure variants
- config / launcher support for `3` structure settings x `2` rewards
- smoke test results
- pilot run outputs
- short Phase 2 status summary
