# Monitoring Workflow

## Scope

This workflow is for the corrected custom scenario only:

- `two_v_two_plus_goalkeepers`

It is intended for repeated inspection during training, not for a broad experiment manager.

## Expected run layout

For one chosen condition, store the seed runs under one root directory:

```text
RUN_ROOT/
  seed_1/
    config.json
    metrics.jsonl
    checkpoints/
  seed_2/
    config.json
    metrics.jsonl
    checkpoints/
  seed_3/
    config.json
    metrics.jsonl
    checkpoints/
```

Example:

```text
Two_V_Two/results/monitoring_runs/r2_progress_shared_ppo/
  seed_1/
  seed_2/
  seed_3/
```

The monitoring scripts fail loudly if:

- a seed run is missing
- `config.json` or `metrics.jsonl` is missing
- the scenario is not `two_v_two_plus_goalkeepers`
- a requested checkpoint cannot be found

## Learning curves

Refresh aggregated learning curves across seeds with:

```bash
Two_V_Two/refresh_learning_curves.sh \
  --run_root Two_V_Two/results/monitoring_runs/r2_progress_shared_ppo \
  --seeds 1 2 3
```

Outputs default to:

```text
Two_V_Two/results/monitoring/<group_name>/curves/
```

For the example above:

```text
Two_V_Two/results/monitoring/r2_progress_shared_ppo/curves/
```

Artifacts:

- `learning_curves.png`
- `curve_summary.json`

Metrics plotted:

- `mean_episode_return`
- `mean_goal_count`
- `mean_pass_count`
- `mean_pass_to_shot_count`
- `mean_assist_count`
- `mean_same_owner_possession_length`

If multiple seeds are present, the figure shows:

- faint per-seed curves
- mean curve
- simple standard-deviation band

## Checkpoint videos

Generate deterministic checkpoint videos across seeds with:

### by target env step

```bash
Two_V_Two/generate_checkpoint_videos.sh \
  --run_root Two_V_Two/results/monitoring_runs/r2_progress_shared_ppo \
  --seeds 1 2 3 \
  --target_env_steps 3200 6400 \
  --episodes 1
```

This resolves each requested env step to the nearest saved checkpoint for each seed.

### by explicit checkpoint filename

```bash
Two_V_Two/generate_checkpoint_videos.sh \
  --run_root Two_V_Two/results/monitoring_runs/r2_progress_shared_ppo \
  --seeds 1 2 3 \
  --checkpoint_names update_000002.pt update_000004.pt \
  --episodes 1
```

Outputs default to:

```text
Two_V_Two/results/monitoring/<group_name>/videos/
```

Artifacts:

- one `.mp4` per requested checkpoint target and seed
- one sidecar `.json` per video from `render_policy_video.py`
- `video_index.json` summarizing requested vs resolved checkpoints

## Operational loop during training

For a chosen condition and three seeds:

1. run training for seeds `1`, `2`, `3`
2. at monitoring points such as `0.5M`, `1M`, `2M`, `5M`:
   - rerun `refresh_learning_curves.sh`
   - rerun `generate_checkpoint_videos.sh` with the target env steps
3. inspect:
   - whether attack structure improves
   - whether teammate-aware behavior appears
   - whether behavior is stable across seeds

## Tested example

The workflow was tested on a corrected-scenario three-seed smoke group stored under:

```text
Two_V_Two/results/monitoring_runs/r2_progress_shared_ppo_smoke/
```

Monitoring outputs were written under:

```text
Two_V_Two/results/monitoring/r2_progress_shared_ppo_smoke/
```
