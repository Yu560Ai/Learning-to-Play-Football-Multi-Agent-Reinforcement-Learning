# Staged Plan for Cooperative MARL in Reduced 2-Agent GRF Attack

## Goal

Study how cooperation emerges in a simplified 2-agent attacking Google Research Football task under controlled changes to reward design and multi-agent structure.

## Environment Summary

Current study setting:

- 2 controlled attacking agents
- built-in left goalkeeper
- 2 built-in right-side outfield opponents
- built-in right goalkeeper
- reduced 55-dimensional local observation
- reduced 14-action discrete space
- cooperative shared reward setup

## Staged Roadmap

### Phase 1: Reward Study With Simple Baseline

- keep the algorithm simple
- use the current shared-policy PPO baseline
- test reward variants:
  - R1: scoring only
  - R2: scoring + progress
  - R3: scoring + progress + assist
  - R4: scoring + progress + assist + anti-selfish penalty
- goal: identify which reward components are most informative for cooperation

### Phase 2: Multi-Agent Structure

- compare:
  - shared PPO baseline
  - shared PPO + agent identity
  - MAPPO with centralized critic
- only carry forward the most informative reward settings from Phase 1

### Phase 3: Training Budget and Stability

- compare fixed final budget results
- plot learning curves over training steps
- run a small budget ablation such as 2M / 5M / 10M steps

### Phase 4: Optional Extras

- entropy tuning
- curriculum
- role bit
- only if needed after Phases 1-3

## Immediate Scope

Only Phase 1 will be implemented now.

Current implementation scope:

- reward implementation for Phase 1
- training scripts/config updates for Phase 1
- Phase 1 runs and saved logs/results

## Deliverables

- `PHASE_PLAN.md`
- reward implementation
- config or training entrypoints for Phase 1
- logs/results for Phase 1 runs
- short summary of what was run
