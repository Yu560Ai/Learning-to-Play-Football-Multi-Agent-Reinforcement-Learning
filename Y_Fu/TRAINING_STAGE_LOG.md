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

## How To Read This File

Use this file in three passes:

1. read `Current Working Conclusion`
2. check `Current Status Summary` and `Stage Passing Criteria`
3. use `Historical Stage Records` only when you need exact evidence or commands

For a shorter failure-focused summary, also read:

- [TRAINING_FAILURE_ARCHIVE.md](/home/yuhan/Codes/RL/Learning-to-Play-Football-Multi-Agent-Reinforcement-Learning/Y_Fu/TRAINING_FAILURE_ARCHIVE.md)

## Current Working Conclusion

Current best interpretation of the logged evidence:

- Academy is still useful as a PPO bootstrap stage
- Academy stage completion has not yet been the same as Academy stage success
- the best current warm-up handoff checkpoint in this log is still `academy_pass_and_shoot_with_keeper/update_10.pt`
- pure `five_vs_five` PPO has already shown a strong negative result at the logged `5M+` and `10M+` checkpoints
- the next meaningful direction is not "just more timesteps", but structured transfer plus better `five_vs_five` objective design

Operationally, this means:

- use Academy to teach primitives
- transfer into `five_vs_five`
- evaluate transfer quality carefully
- treat offline RL as a later `five_vs_five` refinement stage, not as a replacement for curriculum

## Current Budget Reference

Use `env steps` for curriculum budgeting and `agent steps` for interpreting PPO config.

Approximate conversion:

- Stage 1 `academy_run_to_score_with_keeper`: `1 env step = 1 agent step`
- Stage 2 `academy_pass_and_shoot_with_keeper`: `1 env step = 2 agent steps`
- Stage 3 `academy_3_vs_1_with_keeper`: `1 env step = 3 agent steps`
- Stage 4 and Stage 5 `five_vs_five`: `1 env step = 4 agent steps`

Current recommended Academy budget ranges:

- Stage 1 normal range: `250k ~ 400k env steps`
- Stage 2 normal range: `300k ~ 600k env steps`
- Stage 3 normal range: `300k ~ 700k env steps`

Current transfer check for `five_vs_five`:

- early validation at `250k ~ 500k env steps`
- only then commit the long `10M ~ 20M` agent-step block

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
| Stage 3 | `academy_3_vs_1_with_keeper` | Stopped early | Not passed | `2026-04-01 17:10:58` to `17:24:56` | none |
| Stage 4 | `five_vs_five` | In progress | Not evaluated | `2026-04-01 17:25` to active | current run |
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

Video gate:

- pass-to-shot structure appears repeatedly
- the receiver is visibly involved in finishing

### Stage 1: `academy_run_to_score_with_keeper`

Preferred evaluation:

- `20` deterministic episodes across multiple seeds

Pass target:

- scored-episode ratio `>= 0.80`

Support metrics:

- `avg_score_reward >= 0.80`

Video gate:

- direct carry and finishing look deliberate rather than hesitant

### Stage 3: `academy_3_vs_1_with_keeper`

Preferred evaluation:

- `20` deterministic episodes across multiple seeds

Pass target:

- scored-episode ratio `>= 0.55`

Support metrics:

- `avg_score_reward >= 0.55`

Video gate:

- the extra attacker is used meaningfully
- attacks do not always die on the first blocked lane

### Stage 4: `five_vs_five`

Preferred evaluation:

- `20` deterministic episodes across multiple seeds against the built-in bot

Pass target:

- `win_rate >= 0.20`
- `avg_goal_diff >= -0.30`

Support gate:

- passing survives transfer at least occasionally
- shot creation looks cleaner than scratch PPO at the same early budget

### Stage 5: `five_vs_five`

Preferred evaluation:

- `20` deterministic episodes across multiple seeds against the built-in bot

Pass target:

- `win_rate >= 0.55`
- `avg_goal_diff > 0.00`

## Immediate Operational Notes

- current final target remains `five_vs_five`, not `11_vs_11`
- current training still does not use self-play
- stage transitions should be driven by `stage passed`, not merely by `run completed`
- if a run finishes without passing, either extend the budget, tune the setup, or treat it as exploratory only
- if Stage 2 finishes with non-monotonic checkpoint quality, select the transition checkpoint by evaluation, not by latest timestamp
- the best current warm-up candidate from logged evaluations is `update_10.pt`, not `update_110.pt`
- do not start offline RL from `five_vs_five` until early transfer has been checked at roughly `250k ~ 500k env steps`

## Historical Stage Records

The sections below preserve the detailed stage-by-stage evidence, commands, and outputs.

### Stage 0

#### Purpose

- smoke test the new vectorized rollout path
- verify GRF import/runtime path
- measure initial throughput

#### Command

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

#### Timing

Observed checkpoint window:

- first persisted checkpoint: `2026-04-01 14:25:59.151073956 +0800`
- final stage checkpoint (`latest.pt`): `2026-04-01 14:27:28.252382360 +0800`
- observed persisted duration: about `1m 29s`

Checkpoint references:

- [update_2.pt](/home/yuhan/Codes/RL/Learning-to-Play-Football-Multi-Agent-Reinforcement-Learning/Y_Fu/checkpoints/academy_pass_and_shoot_with_keeper/update_2.pt)
- [latest.pt](/home/yuhan/Codes/RL/Learning-to-Play-Football-Multi-Agent-Reinforcement-Learning/Y_Fu/checkpoints/academy_pass_and_shoot_with_keeper/latest.pt)

#### Training Notes

- vectorized rollout ran successfully
- observed training throughput was roughly `env_fps=700~1000`
- observed `samples_per_sec` was roughly `1400~2000`

#### Evaluation

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

#### Representative Video

Video output directory:

- [academy_pass_stage0_seed123](/home/yuhan/Codes/RL/Learning-to-Play-Football-Multi-Agent-Reinforcement-Learning/Y_Fu/videos/stage_log/academy_pass_stage0_seed123)

Representative files:

- [episode_done_20260401-144307765424.avi](/home/yuhan/Codes/RL/Learning-to-Play-Football-Multi-Agent-Reinforcement-Learning/Y_Fu/videos/stage_log/academy_pass_stage0_seed123/episode_done_20260401-144307765424.avi)
- [episode_done_20260401-144307765424.dump](/home/yuhan/Codes/RL/Learning-to-Play-Football-Multi-Agent-Reinforcement-Learning/Y_Fu/videos/stage_log/academy_pass_stage0_seed123/episode_done_20260401-144307765424.dump)

### Stage 1

#### Planned Command

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

#### Status

- not started in this current run log

### Stage 2

#### Purpose

- continue on `academy_pass_and_shoot_with_keeper`
- build passing, supporting run timing, and shot completion
- prepare for `five_vs_five` outfield coordination

#### Command

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

#### Timing

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

#### Evaluation A: Early Stage 2 Checkpoint

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

#### Evaluation B: Newer Stage 2 Checkpoint

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

#### Representative Videos

Early Stage 2 video:

- [academy_pass_stage2_update10_seed123](/home/yuhan/Codes/RL/Learning-to-Play-Football-Multi-Agent-Reinforcement-Learning/Y_Fu/videos/stage_log/academy_pass_stage2_update10_seed123)
- [episode_done_20260401-144307842679.avi](/home/yuhan/Codes/RL/Learning-to-Play-Football-Multi-Agent-Reinforcement-Learning/Y_Fu/videos/stage_log/academy_pass_stage2_update10_seed123/episode_done_20260401-144307842679.avi)

Newer Stage 2 video:

- [academy_pass_stage2_update110_seed123](/home/yuhan/Codes/RL/Learning-to-Play-Football-Multi-Agent-Reinforcement-Learning/Y_Fu/videos/stage_log/academy_pass_stage2_update110_seed123)
- [episode_done_20260401-144533240966.avi](/home/yuhan/Codes/RL/Learning-to-Play-Football-Multi-Agent-Reinforcement-Learning/Y_Fu/videos/stage_log/academy_pass_stage2_update110_seed123/episode_done_20260401-144533240966.avi)

### Stage 3

#### Planned Command

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

#### Planned Evidence To Record

- start and end checkpoint timestamps
- `avg_return`, `avg_goal_diff`, `win_rate`
- one representative video under `Y_Fu/videos/stage_log/`

#### Current Status

- `run stopped early`
- `stage not passed`
- the stage was intentionally cut off because it was consuming environment steps without approaching the pass gate

#### Operational Note

- this stage was not allowed to consume the full budget once it became clear that it was not producing useful progress toward the final `five_vs_five` target

### Stage 4

#### Active Command

```bash
.venv_yfu_grf_sys/bin/python Y_Fu/train.py \
  --preset five_vs_five \
  --num-envs 6 \
  --rollout-steps 256 \
  --total-timesteps 20000000 \
  --save-interval 10 \
  --update-epochs 4 \
  --num-minibatches 1 \
  --init-checkpoint Y_Fu/checkpoints/academy_pass_and_shoot_with_keeper/update_10.pt \
  --seed 42
```

#### Current 5M Checkpoint Note

Observed around:

- `2026-04-01 19:02 +0800`
- roughly `5.4M` agent steps
- around `update 440 / 1628`

Current behavior summary:

- `goals_for` is still effectively `0`
- `success_rate` remains `0.000`
- completed matches are repeatedly full-length `3001` step games
- scorelines are dominated by losses such as `0-1`, `0-2`, `0-3`, `0-4`, `0-5`, `0-6`
- the policy still does not look like meaningful football behavior

Observed training signal:

- training itself is stable
- throughput is roughly `total_sps ≈ 1000`
- environment steps are being consumed normally
- the problem is not runtime stability, but lack of meaningful learning progress

Failure interpretation:

- this is a valuable negative result
- the run has already passed the point where "maybe it just needs a little more time" is a convincing explanation
- earlier experience suggested that around `2M` steps the behavior still looked like nonsense
- at `5M+` steps, that concern is now reinforced rather than weakened
- this strongly suggests that the current setup may be learning the wrong objective or failing to assign credit in a useful way

Operational conclusion:

- continue running to the next hard checkpoint
- if the `10M` checkpoint still shows no meaningful football structure, the next move should be reward revision rather than blindly increasing timesteps

#### Planned Evidence To Record

- checkpoint timestamps
- `five_vs_five` evaluation against the built-in bot
- one representative `five_vs_five` video

#### Current 10M Checkpoint Note

Observed around:

- `2026-04-01 20:19 +0800`
- roughly `10.6M` agent steps
- roughly `2.64M` environment steps
- around `update 861 / 1628`

Current behavior summary:

- `goals_for` remains effectively `0`
- `success_rate` is still repeatedly `0.000`
- completed matches are still dominated by full-length `3001` step games
- scorelines remain dominated by losses such as `0-1`, `0-2`, `0-3`, `0-4`, `0-5`, `0-6`
- behavior still does not look like meaningful `five_vs_five` football

Observed training signal:

- rollout collection remained stable
- throughput remained roughly `total_sps ≈ 1060`
- the run was using environment steps normally
- the failure is about learning quality, not runtime instability

Failure interpretation:

- this is now a high-confidence negative result, not an early noisy checkpoint
- the run is well past the point where "just give it a bit more time" is a strong explanation
- this supports the earlier concern that current reward shaping and credit assignment are not pulling the policy toward real football behavior
- the result is valuable because it narrows the next intervention to reward revision first, then `player_id` only if reward revision still fails

Operational conclusion:

- this run should no longer be treated as evidence that more timesteps alone will solve `five_vs_five`
- the next run should use a reward-only revision so the change remains attributable
- the current run was no longer active at follow-up, so the next step is restart under the revised reward config rather than continue the old setup

### Stage 5

#### Planned Command Template

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

## Related Files

- plan: [THREE_DAY_5V5_TRAINING_PLAN.md](/home/yuhan/Codes/RL/Learning-to-Play-Football-Multi-Agent-Reinforcement-Learning/Y_Fu/THREE_DAY_5V5_TRAINING_PLAN.md)
- log: [TRAINING_STAGE_LOG.md](/home/yuhan/Codes/RL/Learning-to-Play-Football-Multi-Agent-Reinforcement-Learning/Y_Fu/TRAINING_STAGE_LOG.md)
