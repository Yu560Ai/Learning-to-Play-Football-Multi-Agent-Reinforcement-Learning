# Next Steps（短期检查清单）

## 今日（0.5–1 天）
- 重训 5v5 主线（避免旧 ckpt 过滤加载）：`python Y_Yao/train.py --preset five_vs_five --device cpu`
- 每 50k steps 快速评估+录像：`python Y_Yao/evaluate.py --checkpoint Y_Yao/checkpoints/five_vs_five/latest.pt --episodes 3 --device cpu --save-video --video-dir Y_Yao/videos/five_vs_five`
- 监控指标：avg_goal_diff、win_rate、传球成功率/回传次数、gk_out_of_box 计数。
- 若录像出现“控球拖沓/无终结”，先调大 `final_third_entry_reward` 或 `shot_attempt_reward`（小步 0.01）。

## 半天后/明早复盘
- 若推进仍弱：提高 `opponent_attacking_possession_penalty` 或增加“最远持球距离”奖励，抑制后场回传。
- 若 GK 仍出禁区：提升 `gk_out_of_box_penalty` 或仅在无威胁时关闭此惩罚。
- 记录一次对随机策略基线的对比（`--compare-random`）。

## 1 天内方法升级候选
- 试多头 actor（pass/carry/shot/clear 意图分类）或轻量注意力，先在 5v5 验证。
- 考虑 centralized critic / value mixing（TDE）以稳住防守决策。
- 11v11 只在 5v5 指标稳定后再迁移。
