# Next Steps (short horizon)

## Today (0.5–1 day)
- Retrain 5v5 mainline (avoid old ckpt filtered loads): `python Y_Yao/train.py --preset five_vs_five --device cpu`
- Every 50k steps run quick eval + video:  
  `python Y_Yao/evaluate.py --checkpoint Y_Yao/checkpoints/five_vs_five/latest.pt --episodes 3 --device cpu --save-video --video-dir Y_Yao/videos/five_vs_five`
- Track: avg_goal_diff, win_rate, pass success / backward passes, gk_out_of_box counts.
- If “possession but no finish” persists, bump `final_third_entry_reward` or `shot_attempt_reward` in small increments (e.g., +0.01).

## Half-day check / tomorrow morning
- If progression still weak: raise `opponent_attacking_possession_penalty` or add a “max backward distance” bonus/penalty to suppress deep recycling.
- If GK still drifts: increase `gk_out_of_box_penalty` or gate it to only apply under threat.
- Log one random-policy baseline comparison (`--compare-random`).

## Within 1 day – method candidates
- Try multi-head actor (pass/carry/shot/clear intent) or lightweight attention; validate on 5v5 first.
- Consider centralized critic / value mixing (TDE) to stabilize defensive choices.
- Delay 11v11 until 5v5 metrics are solid.
