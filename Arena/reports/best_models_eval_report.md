# Arena Best Models Evaluation

Date: 2026-03-31

## Scope

- Requested evaluation 1: inter-game between the three people in `best_models/`
- Requested evaluation 2: each person's model against the Google built-in baseline

## Model Availability

- `Y_Fu`: model file present
  - `best_models/Y_Fu/shared_policy_ppo_five_vs_five_Y_Fu_0.0%.pt`
- `Y_Yao`: no model `.pt` file present in `best_models/Y_Yao/`
- `X_Jiang`: no model `.pt` file present in `best_models/X_Jiang/`

## Inter-Game Status

Inter-owner matches cannot be completed yet because only one owner folder currently contains a real model checkpoint.

Unavailable pairings:

- `Y_Fu` vs `Y_Yao`: blocked, no model file for `Y_Yao`
- `Y_Fu` vs `X_Jiang`: blocked, no model file for `X_Jiang`
- `Y_Yao` vs `X_Jiang`: blocked, no model file for either side

## Google Baseline Matches

Environment used:

- scenario: `5_vs_5`
- representation: `extracted`
- action set: `v2`
- controlled players: `4` left and `4` right
- channel dimensions: `42 x 42`

### Y_Fu vs Google Built-In

- command:
  - `python -m Arena.run_match --env-name 5_vs_5 --representation extracted --action-set v2 --left-players 4 --right-players 4 --channel-width 42 --channel-height 42 --left-agent yfu_multiagent --left-checkpoint best_models/Y_Fu/shared_policy_ppo_five_vs_five_Y_Fu_0.0%.pt --right-agent google_builtin --episodes 3 --deterministic-left`
- result:
  - `arena_summary: left_avg_reward=0.000 right_avg_reward=0.000 avg_score=0.000-0.000 avg_goal_diff=0.000 left_win_rate=0.000 draw_rate=1.000 avg_length=3001.0`
- interpretation:
  - Y_Fu drew all 3 matches against the Google built-in baseline in this direction.

### Google Built-In vs Y_Fu

- command:
  - `python -m Arena.run_match --env-name 5_vs_5 --representation extracted --action-set v2 --left-players 4 --right-players 4 --channel-width 42 --channel-height 42 --left-agent google_builtin --right-agent yfu_multiagent --right-checkpoint best_models/Y_Fu/shared_policy_ppo_five_vs_five_Y_Fu_0.0%.pt --episodes 3 --deterministic-right`
- result:
  - `arena_summary: left_avg_reward=0.333 right_avg_reward=-0.333 avg_score=0.333-0.000 avg_goal_diff=0.333 left_win_rate=0.333 draw_rate=0.667 avg_length=3001.0`
- interpretation:
  - Google built-in won 1 match and drew 2 when Y_Fu played from the right side in this sample.

## Current Bottom Line

- Full three-person inter-game evaluation is not possible yet because only `Y_Fu` currently has a model file under `best_models/`.
- The current shared `Y_Fu` 5v5 model is competitive enough to draw the Google built-in baseline, but in this small sample it did not beat it.
- In the two evaluated directions combined:
  - `Y_Fu`: 0 wins, 5 draws, 1 loss
  - `Google built-in`: 1 win, 5 draws, 0 losses
