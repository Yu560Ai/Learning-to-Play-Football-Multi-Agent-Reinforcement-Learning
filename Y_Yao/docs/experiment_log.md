# Experiment Log（近期）

- 11v11 shared-policy PPO（无集中 critic）
  - 设置：3/11 控球，简单 MLP，未用 agent_id。
  - 结果：多轮对抗被压制，avg_goal_diff 约 -9，判定为不值得在当前周期继续。
  - 动作：暂停 11v11，资源集中到 5v5。

- 5v5 baseline（无 role embedding，shaping 较弱）
  - 现象：传导/拉开还行，但在被对手压迫时容易回传失误，推进慢。
  - 结论：缺乏角色特异信息和更强进攻 shaping。

- 5v5 强化版（当前主线）
  - 配置：agent_id + role embedding；shaping 如奖励笔记；超参同主线（rollout_steps=512，lr=2.5e-4，ent_coef=0.005）。
  - 评估：旧 ckpt 过滤加载后 avg_goal_diff 仍在负数（约 -3 左右），传控改善但终结缺失，常见回传和被断球。
  - 录像观察：有基本站位与倒脚；攻击推进容易停在前场外沿；GK 偶尔离盒。
  - 动作：重新训练以去除 ckpt/结构不兼容影响；继续调试防守恢复与前插。

备注：录像和原始 log 尚未整理到仓库，保持本地。
