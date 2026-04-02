# Reward Shaping Notes (5v5)

## Passing
- `pass_success_reward = 0.20`: successful pass.
- `pass_progress_reward_scale = 0.10`: forward distance bonus.
- `backward_pass_penalty = 0.02`: clear backward passes.
- `pass_failure_penalty = 0.10`: intercepted or misplaced; self-pass failure included.
- `pass_loose_ball_penalty`: counted in env info for behavior stats.

## Attack progression
- `attacking_possession_reward = 0.002`: sustained possession.
- `final_third_entry_reward = 0.05`: entering attacking third.
- `shot_attempt_reward = 0.03`: shot attempts.

## Defensive recovery
- `possession_recovery_reward = 0.06`: regain possession.
- `defensive_third_recovery_reward = 0.06`: recovery in defensive third.
- `opponent_attacking_possession_penalty = 0.005`: opponent keeps attacking possession.
- `own_half_turnover_penalty = 0.08`: loss in own half.

## GK constraint
- `gk_out_of_box_penalty = 0.02`: GK x > -0.3 while leaving box.

## Notes
- Shaping yields smooth possession but can drift into “no finish”; consider nudging final_third_entry / shot_attempt up instead of adding more terms.
- GK penalty is soft; raise it or gate by threat level if wandering persists.
- Use `evaluate.py` behavior counters (pass_backward, gk_out_of_box, etc.) to quantify changes.
