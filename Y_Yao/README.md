# Y_Yao

Working directory for Google Research Football (multi-agent).

## Current Mainline (2026-04-01)
- Task: `5_vs_5`, 4 controlled players (GK handled by env).
- Algo: shared-policy PPO + agent / role embeddings, CNN encoder, GAE.
- Key hyperparams: rollout_steps=512, update_epochs=3, num_minibatches=2, total_timesteps=500k, learning_rate=2.5e-4, ent_coef=0.005.
- Reward highlights: pass_success +0.20, pass_failure -0.10, pass_progress +0.10, backward_pass -0.02, attacking_possession +0.002, final_third_entry +0.05, recovery +0.06, defensive_third_recovery +0.06, own_half_turnover -0.08, gk_out_of_box -0.02.
- Status: spacing / build-up improved but progression & finishing still weak; old ckpts are not fully shape-compatible with the newer role-embedding model—retrain 5v5 mainline.

## Quick Commands
- Train mainline: `python Y_Yao/train.py --preset five_vs_five --device cpu`
- Evaluate + video: `python Y_Yao/evaluate.py --checkpoint Y_Yao/checkpoints/five_vs_five/latest.pt --episodes 3 --device cpu --save-video --video-dir Y_Yao/videos/five_vs_five`

## Document Map
- `docs/mainline_summary.md`: current 5v5 design, hyperparams, status (start here)
- `docs/experiment_log.md`: recent experiments & failures (recommended)
- `docs/next_steps.md`: half-day / one-day checklist and actions (must read)
- `docs/reward_shaping_notes.md`: active 5v5 reward shaping config and caveats
- `docs/reference_notes.md`: references to other team lines and how this line differs

Read first: `mainline_summary.md`, `experiment_log.md`, `next_steps.md`.
