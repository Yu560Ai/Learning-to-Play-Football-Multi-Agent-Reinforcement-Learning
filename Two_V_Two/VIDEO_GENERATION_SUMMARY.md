# Video Generation Summary

Legacy note:

- the videos described below were generated from academy-based checkpoints created before the environment reset in `Two_V_Two/ENVIRONMENT_RESET.md`
- they should be treated as legacy artifacts, not as evidence for the intended custom `two_v_two_plus_goalkeepers` scenario

## Scripts

- `Two_V_Two/evaluation/render_policy_video.py`
- `Two_V_Two/make_eval_videos.sh`
- `Two_V_Two/make_side_by_side_videos.sh`

## Checkpoints used

### Phase 1

- `R2` baseline:
  - `Two_V_Two/results/phase1_extended/budget_5000000/r2_progress/checkpoints/latest.pt`
  - used as the long-budget stable non-cooperative baseline
- `R3` best checkpoint:
  - `Two_V_Two/results/phase1_extended/budget_5000000/r3_assist/checkpoints/update_000150.pt`
  - selected from `Two_V_Two/results/phase1_extended/r3_checkpoint_sweep/summary.json`
  - this is the best deterministic pass checkpoint from the existing Phase 1 sweep

### Phase 2

- `R3/shared_ppo`:
  - `Two_V_Two/results/phase2_extended/r3_assist/shared_ppo/checkpoints/update_000120.pt`
- `R3/mappo_id_cc`:
  - `Two_V_Two/results/phase2_extended/r3_assist/mappo_id_cc/checkpoints/update_000280.pt`
- source artifact for both:
  - `Two_V_Two/results/phase2_extended/analysis/best_checkpoints.json`
- caveat:
  - the deterministic Phase 2 sweep did not identify any nonzero pass or pass-to-shot checkpoint
  - these were therefore taken as the existing `best_overall` deterministic checkpoints, not as solved cooperative checkpoints

## Generated videos

Saved under `Two_V_Two/results/videos/`.

### Individual videos

- `phase1_r2_progress_baseline.mp4`
- `phase1_r3_assist_best.mp4`
- `phase2_r3_assist_shared_ppo_best.mp4`
- `phase2_r3_assist_mappo_id_cc_best.mp4`

Each individual video has a matching metadata file:

- `phase1_r2_progress_baseline.json`
- `phase1_r3_assist_best.json`
- `phase2_r3_assist_shared_ppo_best.json`
- `phase2_r3_assist_mappo_id_cc_best.json`

### Side-by-side comparison videos

- `phase1_r2_vs_r3_side_by_side.mp4`
- `phase2_r3_shared_vs_mappo_side_by_side.mp4`

## Rendering details

- deterministic action selection
- `1` episode per video
- seed `7`
- output framerate `10` fps
- frames captured directly from the GRF environment via RGB rendering
- on-frame overlay includes reward variant, structure variant, checkpoint, step, return, and pass-related counters

## Notes from generated metadata

- Phase 1 `R2` baseline video:
  - `mean_pass_count = 0.0`
  - `mean_pass_to_shot_count = 0.0`
- Phase 1 `R3` best video:
  - `mean_pass_count = 1.0`
  - `mean_pass_to_shot_count = 0.0`
- Phase 2 `R3/shared_ppo` video:
  - `mean_pass_count = 0.0`
  - `mean_pass_to_shot_count = 0.0`
- Phase 2 `R3/mappo_id_cc` video:
  - `mean_pass_count = 0.0`
  - `mean_pass_to_shot_count = 0.0`

These single-seed videos are meant as visual behavior evidence for already-selected checkpoints, not as replacements for the larger deterministic evaluation summaries.

## Tooling caveat

- the host machine did not have a system `ffmpeg` binary in `PATH`
- `imageio-ffmpeg` was installed into `.venv_yfu_grf_sys` and `Two_V_Two/make_side_by_side_videos.sh` resolves that packaged binary automatically
