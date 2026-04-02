# Y_Yao

Working directory for Google Research Football (multi-agent).

## 当前主线（2026-04-01）
- 任务：`5_vs_5`，4 个可控球员（GK 由环境控制）。
- 算法：shared-policy PPO + agent/role embedding，CNN encoder，GAE。
- 关键超参：rollout_steps=512，update_epochs=3，num_minibatches=2，total_timesteps=500k，learning_rate=2.5e-4，ent_coef=0.005。
- 奖励概要：pass_success +0.20，pass_failure -0.10，pass_progress +0.10，backward_pass -0.02，attacking_possession +0.002，final_third_entry +0.05，recovery +0.06，defensive_third_recovery +0.06，own_half_turnover -0.08，gk_out_of_box -0.02。
- 现状：场面组织有所改善但推进/终结偏弱；旧 ckpt 与最新模型结构（role embedding 等）不完全兼容，需要重新跑 5v5 主线。

## Quick Commands
- 训练主线：`python Y_Yao/train.py --preset five_vs_five --device cpu`
- 评估+录像：`python Y_Yao/evaluate.py --checkpoint Y_Yao/checkpoints/five_vs_five/latest.pt --episodes 3 --device cpu --save-video --video-dir Y_Yao/videos/five_vs_five`

## Document Map
- `docs/mainline_summary.md`：当前 5v5 主线设计、超参与状态（推荐先读）
- `docs/experiment_log.md`：近期实验与失败记录（推荐）
- `docs/next_steps.md`：半天/一天内检查清单与行动路线（强烈推荐）
- `docs/reward_shaping_notes.md`：当前 5v5 reward shaping 配置与注意点
- `docs/reference_notes.md`：参考的他人路线与本线差异

现在最该读：`mainline_summary.md`、`experiment_log.md`、`next_steps.md`。
