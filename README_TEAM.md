# Team Collaboration Guidelines (Google Research Football)

This document captures the workflow and best practices for Y_Fu, Y_Yao, and X_Jiang in this repository.

## 1. What to commit to Git

- ✅ Code (Python scripts, modules, utilities)
- ✅ Configs, experiment definitions, scenario files
- ✅ `scripts/` utilities including `manage_models.py` and `setup_team.ps1`
- ✅ `best_models/` with curated model checkpoints (1-2 per person) only, organized by owner subfolder
- ✅ Documentation (`README*.md`, design notes, issue templates)

- ❌ Large checkpoints outside `best_models/` (`*.pt`, `*.pth`, `*.ckpt` in other dirs)
- ❌ Training dumps, replay videos, logs, intermediate artifacts
- ❌ Virtual environments and local IDE settings

## 2. Model sharing workflow using `scripts/manage_models.py`

1. Train your model in your personal area (e.g. `team_work/<name>/`).
2. Choose one high-quality model and run:

```powershell
python scripts/manage_models.py --add --source <path-to-checkpoint> --name Y_Fu --algo my_algo --scenario my_scenario --winrate 82.3 --notes "best baserline after tuning" --commit
```

3. The script:
   - copies the model to `best_models/<owner>/` with naming `{algo}_{scenario}_{name}_{winrate}%.pt`
   - compresses if >95MB (drop optimizer states if possible)
   - updates `README_TEAM.md` metadata table
   - stages updated files for commit (optionally with `--commit`)

4. Review and push changes.

## 3. Daily git workflow

1. `git checkout main`
2. `git pull --rebase origin main`
3. `git checkout -b feature/<name>-<what>`
4. activate Python env (e.g. `python -m venv .venv` then `.\.venv\Scripts\Activate.ps1`)
5. train/evaluate locally
6. on a strong candidate model, run `manage_models.py` (see above)
7. run unit tests or quick sanity checks
8. `git add .`, `git commit -m "<clear message>"`
9. `git push -u origin feature/<name>-<what>`
10. Open PR, request review from team

## 4. Best practices for collaboration

- Keep branches short-lived and focused on one change.
- Use descriptive commit messages and PR descriptions.
- If you need to store additional large model states, use an external artifact store (Google Cloud Storage / S3 / shared network) and keep references in the repo, not raw binaries.
- Update `README_TEAM.md` when workflow changes.
- Check `git status` often and avoid accidental large files in commits:
  - `git status --short`
  - `git check-ignore -v path/to/file`
- Use `git lfs` if heavy artifacts are necessary (with agreed team policy) and configure `.gitattributes` accordingly.

## 5. Preventing large history bloat

- The repo policy is to keep only `best_models/` for shared checkpoints and ignore all other weights.
- Use owner subfolders under `best_models/`, for example `best_models/Y_Fu/`, `best_models/Y_Yao/`, and `best_models/X_Jiang/`.
- Existing 3.68GB checkpoint history should not grow further by following `.gitignore`.
- If you accidentally commit large checkpoints, use `git filter-repo` or `git rebase -i` to remove them before merge.

## 6. Final 2-Week Project Focus

- The main remaining project target is the multi-agent `five_vs_five` line in `Y_Fu/`.
- Treat `five_vs_five` as the main experiment for training, evaluation, videos, and paper results.
- Treat the smaller `academy_pass_and_shoot_with_keeper` and `academy_3_vs_1_with_keeper` stages only as curriculum or supporting evidence, not the final endpoint.
- De-prioritize `11v11` as a main experiment unless extra time remains after `five_vs_five` is stable.
- De-prioritize side work that does not directly strengthen the `five_vs_five` result, including broad Arena integration work that is not needed for the final evaluation.

Practical rule for the remaining time:

1. First priority: improve and stabilize `Y_Fu/train.py --preset five_vs_five`.
2. Second priority: evaluate `five_vs_five` against random and earlier weak checkpoints using consistent metrics such as `avg_goal_diff`, `win_rate`, and `avg_score_reward`.
3. Third priority: keep only enough curriculum evidence from `2_agents` and `3_agents` to explain how the project reached the `five_vs_five` stage.

Scope rule:

- Do not split the remaining research effort across `2_agents`, `3_agents`, `five_vs_five`, and `11v11` equally.
- The main story for the paper should be: curriculum helped bootstrap behavior, but the final reported multi-agent task is `five_vs_five`.

---

## Model registry section

| File | Algo | Scenario | Owner | Winrate | Notes | Size |
|---|---|---|---|---|---|---|
| [shared_policy_ppo_five_vs_five_Y_Fu_0.0%.pt](best_models/Y_Fu/shared_policy_ppo_five_vs_five_Y_Fu_0.0%.pt) | shared_policy_ppo | five_vs_five | Y_Fu | 0.0% | only shared model for now; representative local 5v5 candidate from evaluation report | 10.56 MB |
