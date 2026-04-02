# Reward Shaping Notes（5v5）

## 传控
- `pass_success_reward = 0.20`：成功传球。
- `pass_progress_reward_scale = 0.10`：向前推进距离奖励。
- `backward_pass_penalty = 0.02`：明显向后回传。
- `pass_failure_penalty = 0.10`：传丢或被断；自传失误包含在内。
- `pass_loose_ball_penalty`：在 env info 中计数，用于行为统计。

## 进攻推进
- `attacking_possession_reward = 0.002`：持续持球。
- `final_third_entry_reward = 0.05`：进入进攻三区。
- `shot_attempt_reward = 0.03`：起脚尝试。

## 防守恢复
- `possession_recovery_reward = 0.06`：抢回球权。
- `defensive_third_recovery_reward = 0.06`：后场抢回。
- `opponent_attacking_possession_penalty = 0.005`：对手持续进攻持球。
- `own_half_turnover_penalty = 0.08`：己方半场丢球。

## GK 约束
- `gk_out_of_box_penalty = 0.02`：GK x 坐标 > -0.3 仍离开禁区。

## 观察提示
- shaping 后控球流畅，但仍需关注“控而不射”；可以逐步上调 `final_third_entry_reward`/`shot_attempt_reward`。
- GK 惩罚仅为软约束，必要时可增大或结合 “无威胁时不罚” 判定。
- 通过 `evaluate.py` 的行为计数（pass_backward, gk_out_of_box 等）来量化调整效果。
