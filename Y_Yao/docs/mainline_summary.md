# 5v5 主线概览（2026-04-01）

## 目标
- 环境：`5_vs_5`，4 名可控球员，GK 由环境控制。
- 叙事：以 shared-policy PPO 为主，强调传控与防守恢复，作为最终 multi-agent five_vs_five 结果的备选线。

## 模型与超参
- 算法：shared-policy PPO，GAE；CNN encoder。
- 嵌入：agent_id embedding=16，role_id embedding=8（角色顺序：LB、RB、CM、ST，num_roles=5）。
- 训练：total_timesteps=500k；rollout_steps=512；update_epochs=3；num_minibatches=2；learning_rate=2.5e-4；ent_coef=0.005。
- 观测：`extracted` 表示，channel_dimensions=(42, 42)。

## 当前奖励 shaping（摘要）
- 进攻：pass_success +0.20；pass_progress_scale +0.10；final_third_entry +0.05；attacking_possession +0.002；shot_attempt +0.03。
- 防守/稳态：possession_recovery +0.06；defensive_third_recovery +0.06。
- 约束：pass_failure -0.10；backward_pass -0.02；own_half_turnover -0.08；gk_out_of_box -0.02；opponent_attacking_possession -0.005。
- 详情见 `reward_shaping_notes.md`。

## 现状与问题
- 旧 ckpt 与最新结构（role embedding、头部尺寸）不完全对齐，严格加载失败，需要 filtered load，性能回落；建议重训。
- 视频观察：站位、出球节奏改善，但推进与终结偏弱，存在回传和被压制；偶见 GK 过线。
- 11v11 线表现差（avg_goal_diff ≈ -9，见实验日志），已暂缓；当前聚焦 5v5。

## 风险与提醒
- 加了较多 shaping，需持续监控是否过拟合“控球但不终结”行为。
- 过滤加载 ckpt 可能掩盖权重形状变化；重训能避免隐性兼容问题。
- 录像/指标务必同步记录 `avg_goal_diff`、`win_rate`、传球成功率、GK 出禁区计数。
