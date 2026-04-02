# Reference Notes（避免重复造轮）

- **Y_Fu 主线**：shared-policy PPO，课程式阶段训练；奖励集中在传控/射门，带 centralized critic。可借鉴：课程阶段切换、best_models 管理。
- **Y_Fu / saltyfish**：11v11 Kaggle 线，重度特征与奖励工程，但偏单人风格；当前不直接复用。
- **X_Jiang**：5v5 role-aware PPO，强化 GK 规则、压迫与两翼推进；可参考其角色定义与前插策略。
- **本线差异**：已加入 agent_id + role embedding、进攻/防守 shaping 更细；计划引入多头 actor 或 centralized critic，但先在 5v5 完成验证。
