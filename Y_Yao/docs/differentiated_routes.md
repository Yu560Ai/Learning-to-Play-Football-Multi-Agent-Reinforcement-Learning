# Differentiated Routes (vs. existing team lines)

## What Y_Fu already covers
- Shared-policy PPO with player_id; heavy reward shaping; reward_v2 shifts weight to final-third entry and shots, minimizes generic possession.
- Action-validity filter: replaces no-ball pass/shot/dribble with idle and aligns PPO to executed actions.
- Academy-first curriculum; five_vs_five runs on top; emphasis on behavior diagnostics.
- Offline RL plan: discrete IQL on five_vs_five replay, then PPO fine-tune; no Academy data mixed in.

## What X_Jiang already covers
- Role-aware PPO for five controllable players (GK, LD, RD, CM, ST).
- Per-role reward shaping for goalkeeper stability, channel defense, pressure, and progression.
- Per-player rollout storage inside a shared 5v5 environment; CPU-friendly PPO pipeline.

## Avoid duplicating
- More generic shared-policy PPO runs with dense shaping.
- Role-aware PPO with per-player shaping only.
- No-ball action filtering and reward_v2-style danger-focused coefficients.
- Offline IQL on five_vs_five replay as a standalone upgrade.

## Y_Yao distinct directions to try
1) **Centralized-attention critic + shared actor**  
   - Keep lightweight shared actor (agent/role embeddings), add an attention-based critic that pools all player embeddings each step (MAPPO-lite).  
   - Target: better credit assignment for support runs and rotations that current shared critic misses.  
   - Quick proto: reuse current PPO buffer; add critic(input = stacked obs + agent/role ids; attention over players); keep actor unchanged.

2) **Intent-layer multi-head actor (options-lite)**  
   - Add a small intent head (pass / carry / shot / clear) producing a discrete option; downstream action head conditioned on intent + role.  
   - Regularize with entropy + KL to avoid single-intent collapse; log intent frequencies for diagnostics.  
   - Goal: separate “what” from “how” without fully separate policies.

3) **Robustness via agent-dropout rollouts**  
   - During rollout, randomly mask one controlled player’s observation/action each few timesteps; critic still sees full state.  
   - Forces policies to maintain shape when a teammate is missing/out-of-position; different from existing curricula.  
   - Measure: possession loss after drop events, recovery time to regain structure.

4) **Event-chain shaping with possession budget**  
   - Instead of rewarding possession itself, allocate a small budget per possession that only pays out on (a) final-third entry, (b) shot, (c) recovery within defensive third after losing ball.  
   - Budget resets on turnover; discourages endless recycling but keeps signal dense enough.  
   - Distinct from reward_v2 because it enforces per-possession accounting, not global dense terms.

## Minimal experiment order
1. Implement centralized-attention critic (single-episode sanity; check loss stability).  
2. Layer intent head; collect intent distribution + pass/shot rates.  
3. Add possession-budget shaping; small ablation vs current shaping.  
4. Only if 1–3 show gains, combine 1+2 in a short 200k-step run before longer 500k.
