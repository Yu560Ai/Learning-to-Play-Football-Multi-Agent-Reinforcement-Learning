# Team Collaboration Guidelines (Google Research Football)

This document captures the workflow and best practices for Y_Fu, Y_Yao, and X_Jiang in this repository.

## 1. What to commit to Git

- ✅ Code (Python scripts, modules, utilities)
- ✅ Configs, experiment definitions, scenario files
- ✅ `scripts/` utilities including `manage_models.py` and `setup_team.ps1`
- ✅ `best_models/` with curated model checkpoints (1-2 per person) only
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
   - copies the model to `best_models/` with naming `{algo}_{scenario}_{name}_{winrate}%.pt`
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
- Existing 3.68GB checkpoint history should not grow further by following `.gitignore`.
- If you accidentally commit large checkpoints, use `git filter-repo` or `git rebase -i` to remove them before merge.

---

## Model registry section

| File | Algo | Scenario | Owner | Winrate | Notes | Size |
|---|---|---|---|---|---|---|
