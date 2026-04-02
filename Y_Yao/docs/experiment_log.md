# Experiment Log (recent)

- 11v11 shared-policy PPO (no centralized critic)  
  - Setup: 3/11 controlled, simple MLP, no agent_id.  
  - Result: repeatedly crushed (avg_goal_diff ≈ -9) → not worth time this cycle.  
  - Action: pause 11v11, focus resources on 5v5.

- 5v5 baseline (no role embedding, weaker shaping)  
  - Observation: circulation OK, but under pressure falls back and loses ball; slow progression.  
  - Conclusion: needs role-specific signal and stronger offensive shaping.

- 5v5 enhanced (current mainline)  
  - Config: agent_id + role embedding; shaping per notes; same core hyperparams (rollout_steps=512, lr=2.5e-4, ent_coef=0.005).  
  - Eval: filtered load of older ckpt gives avg_goal_diff around -3; passing looks better, finishing poor, back-passes common.  
  - Video notes: basic spacing and recycling present; attacks stall near wide channels; GK sometimes leaves box.  
  - Action: retrain from scratch to remove ckpt/shape mismatch; continue tuning defensive recovery and forward runs.

- 5v5 continuation from latest.pt (2026-04-02, +100k steps, cpu)  
  - Setup: resumed from `Y_Yao/checkpoints/five_vs_five/latest.pt`, 49 updates (~100k agent steps), no eval videos yet.  
  - Outcomes: multiple full-length games still 0-x (examples: 0-3, 0-4, 0-2, 0-1, one 0-0); score_reward ≤ 0; success_rate=0.  
  - Failure reasons (hypothesis): plain shared PPO + current shaping is still optimizing for “not losing badly” instead of creating danger; credit assignment across players stays weak; no new signal for shot selection; role embedding alone insufficient.  
  - Action: stop extending this PPO line; switch to differentiated routes (attention critic / intent head / possession-budget shaping) before spending more steps.

Note: raw videos/logs stay local (not in repo).
