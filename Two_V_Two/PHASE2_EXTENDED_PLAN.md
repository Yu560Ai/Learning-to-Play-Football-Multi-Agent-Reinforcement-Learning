# Phase 2 Extended Plan

## Selected Conditions

The focused Phase 2 extension uses exactly these 4 conditions:

- `r2_progress` + `shared_ppo`
- `r2_progress` + `mappo_id_cc`
- `r3_assist` + `shared_ppo`
- `r3_assist` + `mappo_id_cc`

## Why These 4

- `r2_progress/shared_ppo` is the stable non-cooperative baseline.
- `r2_progress/mappo_id_cc` tests whether multi-agent structure alone creates teammate interaction under the control reward.
- `r3_assist/shared_ppo` is the reward-only cooperation candidate from Phase 1.
- `r3_assist/mappo_id_cc` is the main Phase 2 target because the pilot showed the strongest early pass-to-shot signal there.

## Extension Goal

The purpose of this extension is to test whether `r3_assist/mappo_id_cc` retains and strengthens its early pass-to-shot signal under longer training, and whether it separates behaviorally from `r3_assist/shared_ppo` under deterministic checkpoint evaluation.

## Run Plan

- train the 4 selected conditions to `500k` steps
- save checkpoints throughout training under `Two_V_Two/results/phase2_extended/`
- run deterministic evaluation on saved checkpoints, not only `latest.pt`
- compare training curves and checkpoint-level evaluation before making the next Phase 2 decision
