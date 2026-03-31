# Multi-Agent Evaluation Report

- deterministic_policy: `True`
- evaluation_seed_base: `123`
- video_root: `Y_Fu/videos/multiagent`

## 2_agents

- stage: `academy_pass_and_shoot_with_keeper`
- requested_folder_label: `2_agents`

| checkpoint | episodes | avg_score_reward | avg_goal_diff | win_rate | delta_score_vs_random | delta_goal_diff_vs_random | delta_win_rate_vs_random |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `update_5.pt` | 20 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| `update_20.pt` | 20 | 0.000 | 0.000 | 0.000 | 0.050 | 0.050 | 0.000 |
| `update_40.pt` | 20 | 0.000 | 0.000 | 0.000 | 0.050 | 0.050 | 0.000 |

- improvement_vs_first_checkpoint: `avg_score_reward +0.000`, `avg_goal_diff +0.000`, `win_rate +0.000`
- representative_checkpoint: `update_40.pt` (video seed `123`, dir `Y_Fu/videos/multiagent/2_agents`)

## 3_agents

- stage: `academy_3_vs_1_with_keeper`
- requested_folder_label: `3_agents`

| checkpoint | episodes | avg_score_reward | avg_goal_diff | win_rate | delta_score_vs_random | delta_goal_diff_vs_random | delta_win_rate_vs_random |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `update_10.pt` | 20 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| `update_90.pt` | 20 | 0.000 | 0.000 | 0.000 | -0.050 | -0.050 | -0.050 |
| `latest.pt` | 20 | 0.050 | 0.050 | 0.050 | 0.100 | 0.100 | 0.050 |

- improvement_vs_first_checkpoint: `avg_score_reward +0.050`, `avg_goal_diff +0.050`, `win_rate +0.050`
- representative_checkpoint: `latest.pt` (video seed `123`, dir `Y_Fu/videos/multiagent/3_agents`)

## 5_agents

- stage: `five_vs_five`
- requested_folder_label: `5_agents`
- note: Current preset controls 4 players in code, but this folder name follows the requested 5-agent layout.

| checkpoint | episodes | avg_score_reward | avg_goal_diff | win_rate | delta_score_vs_random | delta_goal_diff_vs_random | delta_win_rate_vs_random |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `update_10.pt` | 5 | -0.800 | -0.800 | 0.000 | 2.000 | 2.000 | 0.000 |
| `update_140.pt` | 5 | -1.400 | -1.400 | 0.000 | 0.600 | 0.600 | 0.000 |
| `latest.pt` | 5 | -3.000 | -3.000 | 0.000 | 0.000 | 0.000 | 0.000 |

- improvement_vs_first_checkpoint: `avg_score_reward +0.000`, `avg_goal_diff +0.000`, `win_rate +0.000`
- representative_checkpoint: `update_10.pt` (video seed `123`, dir `Y_Fu/videos/multiagent/5_agents`)

## 11_agents

- stage: `full_11v11_residual`
- requested_folder_label: `11_agents`

| checkpoint | episodes | avg_score_reward | avg_goal_diff | win_rate | delta_score_vs_random | delta_goal_diff_vs_random | delta_win_rate_vs_random |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `update_20.pt` | 5 | -14.800 | -14.800 | 0.000 | -7.000 | -7.000 | 0.000 |
| `update_80.pt` | 5 | -6.800 | -6.800 | 0.000 | 0.600 | 0.600 | 0.000 |
| `latest.pt` | 5 | -15.400 | -15.400 | 0.000 | -7.800 | -7.800 | 0.000 |

- improvement_vs_first_checkpoint: `avg_score_reward +8.000`, `avg_goal_diff +8.000`, `win_rate +0.000`
- representative_checkpoint: `update_80.pt` (video seed `123`, dir `Y_Fu/videos/multiagent/11_agents`)

