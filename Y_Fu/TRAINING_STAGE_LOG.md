# Training Stage Log

## Scope

This log tracks the current `Y_Fu` 3-day training plan for the final target:

- `5_vs_5` multi-agent outfield player training
- shared-policy PPO
- parallel environment sampling
- no self-play yet

Current opponent source:

- `academy_*`: GRF scripted teaching scenarios
- `five_vs_five`: GRF built-in scripted bot

This file is intended to be precise and reproducible:

- exact commands are recorded
- checkpoint timestamps are recorded
- evaluation outputs are recorded
- representative video paths are recorded when available

## Status Definitions

This log uses two separate judgments for each stage.

### `run completed`

This means:

- the configured training command reached its budget
- checkpoints were produced
- the process exited normally

It does not mean the policy solved the task.

### `stage passed`

This means:

- the stage-specific evaluation threshold was met
- the checkpoint is acceptable as a real handoff into the next stage

This is the meaningful success signal.

## Evaluation Protocol

Current logged evaluation is a reproducibility check, not a broad generalization benchmark.

Protocol used:

- `--seed 123`
- `--deterministic`
- `--device cpu`
- `--compare-random` when measuring against a random baseline

Important consequence:

- deterministic evaluation with a fixed seed can replay nearly the same episode multiple times
- this is useful for checkpoint-to-checkpoint comparison
- this is not enough to estimate full robustness

Later, for broader evaluation, add:

- multiple seeds
- non-deterministic action sampling or varied environment seeds
- direct `five_vs_five` evaluation after warm-up

## Current Status Summary

| Stage | Preset | Run Status | Pass Status | Evidence Window | Preferred Checkpoint |
|---|---|---|---|---|---|
| Stage 0 | `academy_pass_and_shoot_with_keeper` | Completed | Not passed | `2026-04-01 14:25:59` to `14:27:28` | `latest.pt` |
| Stage 1 | `academy_run_to_score_with_keeper` | Not started | Not evaluated | n/a | n/a |
| Stage 2 | `academy_pass_and_shoot_with_keeper` | Completed | Not passed | `2026-04-01 14:38:39` to `14:53:03` | `update_10.pt` |
| Stage 3 | `academy_3_vs_1_with_keeper` | In progress | Not evaluated | `2026-04-01 17:10:58` to active | pending |
| Stage 4 | `five_vs_five` | Not started | Not evaluated | n/a | n/a |
| Stage 5 | `five_vs_five` | Not started | Not evaluated | n/a | n/a |

## Stage Passing Criteria

These gates decide whether a stage is actually finished in the learning sense.

### Stage 0 and Stage 2: `academy_pass_and_shoot_with_keeper`

Preferred evaluation:

- `20` deterministic episodes across multiple seeds, or
- `50` total episodes if doing a single-seed sweep

Pass target:

- scored-episode ratio `>= 0.60`

Support metrics:

- `avg_score_reward >= 0.60`
- `win_rate >= 0.60`

### Stage 1: `academy_run_to_score_with_keeper`

Preferred evaluation:

- `20` deterministic episodes across multiple seeds

Pass target:

- scored-episode ratio `>= 0.80`

Support metrics:

- `avg_score_reward >= 0.80`

### Stage 3: `academy_3_vs_1_with_keeper`

Preferred evaluation:

- `20` deterministic episodes across multiple seeds

Pass target:

- scored-episode ratio `>= 0.55`

Support metrics:

- `avg_score_reward >= 0.55`

### Stage 4: `five_vs_five`

Preferred evaluation:

- `20` deterministic episodes across multiple seeds against the built-in bot

Pass target:

- `win_rate >= 0.35`
- `avg_goal_diff >= -0.10`

### Stage 5: `five_vs_five`

Preferred evaluation:

- `20` deterministic episodes across multiple seeds against the built-in bot

Pass target:

- `win_rate >= 0.55`
- `avg_goal_diff > 0.00`

## Stage 0

### Purpose

- smoke test the new vectorized rollout path
- verify GRF import/runtime path
- measure initial throughput

### Command

```bash
.venv_yfu_grf_sys/bin/python Y_Fu/train.py \
  --preset academy_pass_and_shoot_with_keeper \
  --num-envs 6 \
  --rollout-steps 128 \
  --total-timesteps 80000 \
  --save-interval 2 \
  --update-epochs 4 \
  --num-minibatches 1 \
  --seed 42
```

### Timing

Observed checkpoint window:

- first persisted checkpoint: `2026-04-01 14:25:59.151073956 +0800`
- final stage checkpoint (`latest.pt`): `2026-04-01 14:27:28.252382360 +0800`
- observed persisted duration: about `1m 29s`

Checkpoint references:

- [update_2.pt](/home/yuhan/Codes/RL/Learning-to-Play-Football-Multi-Agent-Reinforcement-Learning/Y_Fu/checkpoints/academy_pass_and_shoot_with_keeper/update_2.pt)
- [latest.pt](/home/yuhan/Codes/RL/Learning-to-Play-Football-Multi-Agent-Reinforcement-Learning/Y_Fu/checkpoints/academy_pass_and_shoot_with_keeper/latest.pt)

### Training Notes

- vectorized rollout ran successfully
- observed training throughput was roughly `env_fps=700~1000`
- observed `samples_per_sec` was roughly `1400~2000`

### Evaluation

Checkpoint:

- [latest.pt](/home/yuhan/Codes/RL/Learning-to-Play-Football-Multi-Agent-Reinforcement-Learning/Y_Fu/checkpoints/academy_pass_and_shoot_with_keeper/latest.pt)

Command:

```bash
.venv_yfu_grf_sys/bin/python Y_Fu/evaluate.py \
  --checkpoint Y_Fu/checkpoints/academy_pass_and_shoot_with_keeper/latest.pt \
  --episodes 5 \
  --deterministic \
  --compare-random \
  --device cpu \
  --seed 123
```

Result:

- trained policy: `avg_return=0.790 avg_score_reward=0.000 avg_goal_diff=0.000 win_rate=0.000 avg_length=186.0`
- random policy: `avg_return=0.502 avg_score_reward=0.200 avg_goal_diff=0.200 win_rate=0.200 avg_length=191.4`
- delta vs random: `avg_return=+0.288 avg_score_reward=-0.200 avg_goal_diff=-0.200 win_rate=-0.200`

Interpretation:

- the policy improved shaped return
- it still was not reliably converting the scenario into goals
- this is acceptable for a smoke-test stage

Verdict:

- `run completed`
- `stage not passed`

### Representative Video

Video output directory:

- [academy_pass_stage0_seed123](/home/yuhan/Codes/RL/Learning-to-Play-Football-Multi-Agent-Reinforcement-Learning/Y_Fu/videos/stage_log/academy_pass_stage0_seed123)

Representative files:

- [episode_done_20260401-144307765424.avi](/home/yuhan/Codes/RL/Learning-to-Play-Football-Multi-Agent-Reinforcement-Learning/Y_Fu/videos/stage_log/academy_pass_stage0_seed123/episode_done_20260401-144307765424.avi)
- [episode_done_20260401-144307765424.dump](/home/yuhan/Codes/RL/Learning-to-Play-Football-Multi-Agent-Reinforcement-Learning/Y_Fu/videos/stage_log/academy_pass_stage0_seed123/episode_done_20260401-144307765424.dump)

## Stage 1

### Planned Command

```bash
.venv_yfu_grf_sys/bin/python Y_Fu/train.py \
  --preset academy_run_to_score_with_keeper \
  --num-envs 8 \
  --rollout-steps 128 \
  --total-timesteps 300000 \
  --save-interval 5 \
  --update-epochs 4 \
  --num-minibatches 1 \
  --seed 42
```

### Status

- not started in this current run log

## Stage 2

### Purpose

- continue on `academy_pass_and_shoot_with_keeper`
- build passing, supporting run timing, and shot completion
- prepare for `five_vs_five` outfield coordination

### Command

```bash
.venv_yfu_grf_sys/bin/python Y_Fu/train.py \
  --preset academy_pass_and_shoot_with_keeper \
  --num-envs 8 \
  --rollout-steps 192 \
  --total-timesteps 800000 \
  --save-interval 5 \
  --update-epochs 4 \
  --num-minibatches 1 \
  --init-checkpoint Y_Fu/checkpoints/academy_pass_and_shoot_with_keeper/latest.pt \
  --seed 42
```

### Timing

Observed persisted checkpoint window:

- first persisted checkpoint in this run: `2026-04-01 14:38:39.896829967 +0800` for `update_5.pt`
- final stage checkpoint (`latest.pt`): `2026-04-01 14:53:03.220798256 +0800`
- observed persisted duration through stage completion: about `14m 23s`

Checkpoint references:

- [update_10.pt](/home/yuhan/Codes/RL/Learning-to-Play-Football-Multi-Agent-Reinforcement-Learning/Y_Fu/checkpoints/academy_pass_and_shoot_with_keeper/update_10.pt)
- [update_110.pt](/home/yuhan/Codes/RL/Learning-to-Play-Football-Multi-Agent-Reinforcement-Learning/Y_Fu/checkpoints/academy_pass_and_shoot_with_keeper/update_110.pt)

Completion snapshot:

- training reached `update 261/261` and exited normally
- current training logs were still dominated by `0-0` outcomes
- throughput was roughly `env_fps=670~1190`
- `samples_per_sec` was roughly `1345~2380`

### Evaluation A: Early Stage 2 Checkpoint

Checkpoint:

- [update_10.pt](/home/yuhan/Codes/RL/Learning-to-Play-Football-Multi-Agent-Reinforcement-Learning/Y_Fu/checkpoints/academy_pass_and_shoot_with_keeper/update_10.pt)

Command:

```bash
.venv_yfu_grf_sys/bin/python Y_Fu/evaluate.py \
  --checkpoint Y_Fu/checkpoints/academy_pass_and_shoot_with_keeper/update_10.pt \
  --episodes 5 \
  --deterministic \
  --compare-random \
  --device cpu \
  --seed 123
```

Result:

- trained policy: `avg_return=1.558 avg_score_reward=0.000 avg_goal_diff=0.000 win_rate=0.000 avg_length=401.0`
- random policy: `avg_return=0.752 avg_score_reward=0.200 avg_goal_diff=0.200 win_rate=0.200 avg_length=111.4`
- delta vs random: `avg_return=+0.806 avg_score_reward=-0.200 avg_goal_diff=-0.200 win_rate=-0.200`

Interpretation:

- the checkpoint strongly improved shaped return
- but it still failed to actually score in the deterministic test episodes
- this suggests improved possession/progression behavior without finish reliability

### Evaluation B: Newer Stage 2 Checkpoint

Checkpoint:

- [update_110.pt](/home/yuhan/Codes/RL/Learning-to-Play-Football-Multi-Agent-Reinforcement-Learning/Y_Fu/checkpoints/academy_pass_and_shoot_with_keeper/update_110.pt)

Command:

```bash
.venv_yfu_grf_sys/bin/python Y_Fu/evaluate.py \
  --checkpoint Y_Fu/checkpoints/academy_pass_and_shoot_with_keeper/update_110.pt \
  --episodes 5 \
  --deterministic \
  --compare-random \
  --device cpu \
  --seed 123
```

Result:

- trained policy: `avg_return=0.711 avg_score_reward=0.000 avg_goal_diff=0.000 win_rate=0.000 avg_length=276.0`
- random policy: `avg_return=0.846 avg_score_reward=0.200 avg_goal_diff=0.200 win_rate=0.200 avg_length=124.4`
- delta vs random: `avg_return=-0.135 avg_score_reward=-0.200 avg_goal_diff=-0.200 win_rate=-0.200`

Interpretation:

- checkpoint quality is not monotonic
- at least on the fixed-seed deterministic test, `update_110` is worse than `update_10`
- this is exactly why stage logging should keep multiple candidate checkpoints instead of assuming the latest one is best

Verdict:

- `run completed`
- `stage not passed`
- preferred handoff checkpoint: `update_10.pt`

### Representative Videos

Early Stage 2 video:

- [academy_pass_stage2_update10_seed123](/home/yuhan/Codes/RL/Learning-to-Play-Football-Multi-Agent-Reinforcement-Learning/Y_Fu/videos/stage_log/academy_pass_stage2_update10_seed123)
- [episode_done_20260401-144307842679.avi](/home/yuhan/Codes/RL/Learning-to-Play-Football-Multi-Agent-Reinforcement-Learning/Y_Fu/videos/stage_log/academy_pass_stage2_update10_seed123/episode_done_20260401-144307842679.avi)

Newer Stage 2 video:

- [academy_pass_stage2_update110_seed123](/home/yuhan/Codes/RL/Learning-to-Play-Football-Multi-Agent-Reinforcement-Learning/Y_Fu/videos/stage_log/academy_pass_stage2_update110_seed123)
- [episode_done_20260401-144533240966.avi](/home/yuhan/Codes/RL/Learning-to-Play-Football-Multi-Agent-Reinforcement-Learning/Y_Fu/videos/stage_log/academy_pass_stage2_update110_seed123/episode_done_20260401-144533240966.avi)

## Stage 3

### Planned Command

```bash
.venv_yfu_grf_sys/bin/python Y_Fu/train.py \
  --preset academy_3_vs_1_with_keeper \
  --num-envs 6 \
  --rollout-steps 192 \
  --total-timesteps 1200000 \
  --save-interval 5 \
  --update-epochs 4 \
  --num-minibatches 1 \
  --seed 42
```

### Planned Evidence To Record

- start and end checkpoint timestamps
- `avg_return`, `avg_goal_diff`, `win_rate`
- one representative video under `Y_Fu/videos/stage_log/`

### Current Status

- `run in progress`
- `stage not yet evaluated`
- this stage should not unlock `five_vs_five` automatically just because the configured timesteps finish

## Stage 4

### Planned Command Template

```bash
.venv_yfu_grf_sys/bin/python Y_Fu/train.py \
  --preset five_vs_five \
  --num-envs 6 \
  --rollout-steps 256 \
  --total-timesteps 10000000 \
  --save-interval 10 \
  --update-epochs 4 \
  --num-minibatches 1 \
  --init-checkpoint <best_warmup_checkpoint> \
  --seed 42
```

### Planned Evidence To Record

- checkpoint timestamps
- `five_vs_five` evaluation against the built-in bot
- one representative `five_vs_five` video

## Stage 5

### Planned Command Template

```bash
.venv_yfu_grf_sys/bin/python Y_Fu/train.py \
  --preset five_vs_five \
  --num-envs 6 \
  --rollout-steps 256 \
  --total-timesteps 10000000 \
  --save-interval 10 \
  --update-epochs 4 \
  --num-minibatches 1 \
  --init-checkpoint <best_stage4_checkpoint> \
  --seed 42
```

Optional scale-up if stable:

- raise `total_timesteps` toward `40_000_000`
- test `num_envs=8` only if throughput improves without instability

## Immediate Operational Notes

- current final target remains `five_vs_five`, not `11_vs_11`
- current training still does not use self-play
- stage transitions should be driven by `stage passed`, not merely by `run completed`
- if a run finishes without passing, either extend the budget, tune the setup, or treat it as exploratory only
- if Stage 2 finishes with non-monotonic checkpoint quality, select the transition checkpoint by evaluation, not by latest timestamp
- the best current warm-up candidate from logged evaluations is `update_10.pt`, not `update_110.pt`

## Related Files

- plan: [THREE_DAY_5V5_TRAINING_PLAN.md](/home/yuhan/Codes/RL/Learning-to-Play-Football-Multi-Agent-Reinforcement-Learning/Y_Fu/THREE_DAY_5V5_TRAINING_PLAN.md)
- log: [TRAINING_STAGE_LOG.md](/home/yuhan/Codes/RL/Learning-to-Play-Football-Multi-Agent-Reinforcement-Learning/Y_Fu/TRAINING_STAGE_LOG.md)
