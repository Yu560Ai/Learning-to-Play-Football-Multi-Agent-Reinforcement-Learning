# 5v5 Mainline Snapshot (2026-04-01)

## Goal
- Environment: `5_vs_5`, 4 controlled players, GK handled by env.
- Narrative: shared-policy PPO focused on passing and defensive recovery; candidate line for final multi-agent five_vs_five result.

## Model & Hyperparams
- Algorithm: shared-policy PPO with GAE; CNN encoder.
- Embeddings: agent_id dim 16; role_id dim 8 (roles: LB, RB, CM, ST; num_roles=5).
- Training: total_timesteps=500k; rollout_steps=512; update_epochs=3; num_minibatches=2; learning_rate=2.5e-4; ent_coef=0.005.
- Observation: `extracted`, channel_dimensions=(42, 42).

## Current Reward Shaping (summary)
- Offense: pass_success +0.20; pass_progress_scale +0.10; final_third_entry +0.05; attacking_possession +0.002; shot_attempt +0.03.
- Defense / stability: possession_recovery +0.06; defensive_third_recovery +0.06.
- Constraints: pass_failure -0.10; backward_pass -0.02; own_half_turnover -0.08; gk_out_of_box -0.02; opponent_attacking_possession -0.005.
- Details in `reward_shaping_notes.md`.

## Status & Issues
- Older ckpts mismatch newer structure (role embedding, head sizes); strict load fails, filtered load works but hurts performance → retrain.
- Video review: spacing and circulation better; progression/finishing weak; frequent back-passes and turnovers; GK occasionally leaves the box.
- 11v11 line performs poorly (avg_goal_diff ≈ -9, see experiment log); parked for now; focus is 5v5.

## Risks & Reminders
- Heavy shaping—watch for “possession without finishing”; consider gradually boosting final_third_entry / shot_attempt instead of piling more shaping.
- Filtered ckpt loads can hide structural drift; fresh training avoids silent shape issues.
- Always log `avg_goal_diff`, `win_rate`, pass success rate, and `gk_out_of_box` counts in eval videos.
