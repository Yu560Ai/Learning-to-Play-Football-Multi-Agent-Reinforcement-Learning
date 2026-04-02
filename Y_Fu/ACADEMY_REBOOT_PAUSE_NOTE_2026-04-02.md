# Academy Reboot Pause Note 2026-04-02

## Status

Training is intentionally paused for the day.

There is currently:

- no active `Y_Fu/train.py` process
- no active `Y_Fu/evaluate.py` process

## What Was Finished Before Pausing

### 1. No-ball action filter fix

Completed:

- non-ball players can no longer execute `pass` / `shot` / `dribble` as real environment actions
- invalid ball-only actions are replaced with `idle`
- PPO rollout now stores executed actions, not just sampled actions
- new diagnostics were added:
  - `invalid_ball_skill_rate`
  - `invalid_no_ball_pass_rate`
  - `invalid_no_ball_shot_rate`

Related doc:

- [NO_BALL_ACTION_FILTER_PLAN.md](/home/yuhan/Codes/RL/Learning-to-Play-Football-Multi-Agent-Reinforcement-Learning/Y_Fu/NO_BALL_ACTION_FILTER_PLAN.md)

### 2. Simple tests

Completed:

- `py_compile` check passed for the updated PPO / env files
- helper smoke test passed for no-ball action sanitization
- a 1-update GRF smoke run passed and printed the new invalid-action metrics

## Academy Reboot Attempt

The new direction is:

- pause the failing `five_vs_five` PPO line
- return to Academy
- use `academy_pass_and_shoot_with_keeper` as the controlled reboot task

Related doc:

- [ACADEMY_REBOOT_PLAN.md](/home/yuhan/Codes/RL/Learning-to-Play-Football-Multi-Agent-Reinforcement-Learning/Y_Fu/ACADEMY_REBOOT_PLAN.md)

## Saved Resume Point

Checkpoint:

- [update_5.pt](/home/yuhan/Codes/RL/Learning-to-Play-Football-Multi-Agent-Reinforcement-Learning/Y_Fu/checkpoints/academy_pass_reboot_v1/update_5.pt)

Checkpoint metadata:

- `update = 5`
- `total_agent_steps = 7680`
- `total_env_steps = 3840`
- `num_envs = 4`
- `num_players = 2`

Interpretation:

- the multi-env Academy reboot did start successfully
- at least one real checkpoint was written
- this is the cleanest local resume point for the next session

## Recommended Restart Command

Use this command to resume from the saved reboot checkpoint:

```bash
.venv_yfu_grf_sys/bin/python Y_Fu/train.py \
  --preset academy_pass_and_shoot_with_keeper \
  --use-player-id \
  --num-envs 4 \
  --rollout-steps 192 \
  --total-timesteps 400000 \
  --save-interval 5 \
  --update-epochs 4 \
  --num-minibatches 1 \
  --device cpu \
  --seed 42 \
  --save-dir Y_Fu/checkpoints/academy_pass_reboot_v1 \
  --logdir Y_Fu/logs/academy_pass_reboot_v1 \
  --init-checkpoint Y_Fu/checkpoints/academy_pass_reboot_v1/update_5.pt
```

## First Things To Check After Restart

At the next session, check these first:

- does the run continue to write checkpoints normally
- does `invalid_no_ball_pass_rate` start dropping
- do videos start showing real pass-then-shoot structure
- does the Academy stage look solvable before increasing task complexity again
