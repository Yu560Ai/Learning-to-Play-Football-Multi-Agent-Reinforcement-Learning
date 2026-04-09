# Roadmap

This is the working roadmap for `Two_V_Two`.

It keeps the useful structure from the earlier GPT outline, but adjusts the first steps to match the actual public TiKick repository and the code already built in this folder.

## What We Keep

- TiKick is the training backbone reference
- TiZero is a tooling and evaluation reference, not the main trainer
- start with a minimal runnable baseline before shrinking to the full paper spec
- add probes before adding many algorithm variants
- run a very small first experiment matrix

## What We Change

- we do not wait to "run TiKick end to end once" because the public TiKick release does not expose a public football training runner
- instead, we already vendor TiKick’s useful public pieces and run a local trainer built on top of them
- TiZero remains optional until replay and evaluation tooling become the bottleneck

## Ordered Plan

### Phase 0

Freeze the study setting:

- `2` attacking RL agents
- built-in goalkeeper
- built-in defenders
- shared policy
- one main training environment
- academy tasks only for probing

### Phase 1

Get the local baseline running end to end:

- local training entry point
- local GRF wrapper
- checkpoint save and load
- smoke-tested PPO update loop

Current status:

- done

### Phase 2

Stabilize the training backbone:

- keep TiKick-derived policy and buffer pieces
- keep only the code paths needed for shared-policy PPO training
- do not expand scope into full-game or distributed training

### Phase 3

Build the exact custom `2`-agent environment interface:

- keep the same training scenario at first
- keep the original observation and action interface until the wrapper is stable
- isolate environment changes from algorithm changes

### Phase 4

Shrink to the paper setting:

- replace the observation with the reduced local vector
- replace the action space with the reduced discrete action set
- keep parameter sharing

### Phase 5

Add the evaluation layer:

- academy probes
- goal rate logging
- return curves
- replay dumps
- action frequency statistics

### Phase 6

Run exactly three first experiments:

- PPO baseline
- PPO + KL prior
- PPO + reward shaping

### Phase 7

Use replay-driven refinement:

- inspect failures
- add one change at a time
- compare with the same evaluation suite

### Phase 8

Produce final behavior-emergence outputs:

- return vs steps
- goal success vs steps
- academy probe curves
- action distributions over time
- cooperation metrics

## Immediate Priority

1. Keep the current baseline stable.
2. Freeze the reduced action spec.
3. Freeze the `55`-dim observation spec.
4. Add evaluation probes to the current trainer.
5. Run the first plain PPO baseline experiment.
