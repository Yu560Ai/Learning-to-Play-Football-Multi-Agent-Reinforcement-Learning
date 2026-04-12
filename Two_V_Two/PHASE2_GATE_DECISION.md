# Phase 2 Gate Decision

## Decision

Option A

`R3 is sufficiently distinct from R2; proceed to Phase 2 with R2 and R3.`

## Reasoning

- `R2` remains the stable non-cooperative baseline. Its deterministic eval stays at zero passes and zero assists.
- The checkpoint sweep for `R3` changed the decision:
  - saved sweep results: `Two_V_Two/results/phase1_extended/r3_checkpoint_sweep/sweep_results.json`
  - compact ranking: `Two_V_Two/results/phase1_extended/r3_checkpoint_sweep/top_checkpoints.md`
  - sweep summary: `Two_V_Two/results/phase1_extended/r3_checkpoint_sweep/summary.json`
- Across `88` deterministic checkpoint evaluations, `R3` produced `7` checkpoints with nonzero `pass_count`.
- Best checkpoint found:
  - run: `5M`
  - checkpoint: `update_000150.pt`
  - env steps: `240000`
  - `mean_pass_count = 0.95`
  - `mean_pass_to_shot_count = 0.0`
  - `mean_assist_count = 0.0`
- Additional deterministic pass checkpoints also appeared at:
  - `1M update_000525.pt` with `mean_pass_count = 0.20`
  - `5M update_001600.pt` with `mean_pass_count = 0.15`
  - `5M update_001750.pt` with `mean_pass_count = 0.20`

This is enough to establish that `R3` is behaviorally distinct from `R2` under deterministic evaluation, even though the distinction is still limited to passing and has not yet matured into pass-to-shot or assist behavior.

## Practical Interpretation

- `R3` is not a solved cooperative reward.
- But it now clears the actual Phase 2 gate:
  - nonzero deterministic pass-related behavior
  - meaningful separation from `R2`
  - teammate-aware structure worth testing under identity and centralized-critic changes
- The correct Phase 2 framing is not "R3 already solved cooperation."
- It is "R3 induces partial cooperative structure that Phase 2 may stabilize or strengthen."

## Immediate Next Step

Proceed to Phase 2 with:

- `R2` as the stable baseline reward
- `R3` as the cooperative-structure candidate reward

Important caveat:

- for `R3`, checkpoint selection matters; `latest` is not the best behavioral summary
- Phase 2 comparisons should keep that in mind when interpreting outcomes
