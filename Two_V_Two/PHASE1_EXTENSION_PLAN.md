# Phase 1 Extension Plan

## Why 200k Is Too Early

The 200k-step Phase 1 pilot was enough to separate sparse and dense reward behavior, but it is still too early to judge late-activating cooperative rewards.

- `R2` can improve quickly because progress reward is dense.
- `R3` depends on successful teammate passes followed by later scoring, which is rare early in training.
- `R4` can change behavior earlier through penalty pressure, but that does not guarantee stable cooperative outcomes.

So weak `R3` or noisy `R4` behavior at 200k should not be treated as a final verdict.

## Expected Behavior At Larger Budgets

Expected around `1M` steps:

- `R1`: still weak sparse reference
- `R2`: strongest early-learning baseline for return and optimization stability
- `R3`: may begin to separate from `R2` on pass-related behavior and possibly first assist events
- `R4`: may reduce possession streaks and increase pass pressure, but with more instability

If needed, extend to `2M` or `5M` for the same comparison before drawing final Phase 1 conclusions.

## Next-Step Plan

- extend Phase 1 runs beyond the current pilot
- prioritize `R2`, `R3`, and `R4`
- keep `R1` as a sparse reference when practical
- reuse the current shared-policy PPO pipeline
- generate longer learning curves
- add budget comparison plots across completed checkpoints
- use deterministic evaluation at completed budgets before making conclusions
