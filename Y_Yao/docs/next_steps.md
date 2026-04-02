# Next Steps (short horizon)

## Immediate (today, ~0.5–1 day)
- Do not extend plain PPO until new ideas are tried; last +100k steps showed 0 goals and full-length draws/losses.
- Run evaluation on latest checkpoint with video + random baseline:  
  `python Y_Yao/evaluate.py --checkpoint Y_Yao/checkpoints/five_vs_five/latest.pt --episodes 3 --device cpu --save-video --video-dir Y_Yao/videos/five_vs_five --compare-random`
- Log: avg_goal_diff, win_rate, pass success/backward count, gk_out_of_box, intent-like action mix (pass/carry/shot/clear if available).

## Short experiments to start (pick one first)
1) **Centralized-attention critic (shared actor unchanged)**  
   - Add critic that attends over player embeddings each step; keep actor/head as-is.  
   - Metric: win_rate, goal_diff vs random after 50k.
2) **Intent-layer actor**  
   - Add discrete intent head (pass/carry/shot/clear); condition action head on intent+role.  
   - Track intent distribution; prevent collapse via entropy/KL.
3) **Possession-budget shaping**  
   - Per-possession budget paying only on final-third entry / shot / defensive-third recovery; resets on turnover.  
   - Goal: kill endless recycling without over-penalizing buildup.

## If time remains (tomorrow)
- Try agent-dropout rollouts (mask one player intermittently) to harden shape; measure recovery time to regain possession/structure.
- Re-evaluate with videos after each 50k block; keep random baseline comparison.

## Defer / avoid
- More plain PPO with current shaping (low return on compute).
- 11v11 until 5v5 shows clear improvement with the new methods.
