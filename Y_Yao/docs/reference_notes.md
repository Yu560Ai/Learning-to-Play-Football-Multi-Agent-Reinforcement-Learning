# Reference Notes (to avoid duplicate work)

- **Y_Fu mainline**: shared-policy PPO with curriculum stages; shaping centered on passing/shooting; uses centralized critic. Borrow: staged curriculum, best_models workflow.
- **Y_Fu / saltyfish**: 11v11 Kaggle-style line with heavy feature/reward engineering; single-user style; not directly reused here.
- **X_Jiang**: 5v5 role-aware PPO, strong GK rules, pressing and wide progression; consider their role definitions and forward runs.
- **This line**: already uses agent_id + role embedding plus finer attack/defense shaping; planned upgrades are multi-head actor or centralized critic, validated on 5v5 first.
