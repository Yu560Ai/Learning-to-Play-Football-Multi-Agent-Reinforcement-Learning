# Y_Fu Football RL Baseline

This folder contains a self-contained **PyTorch PPO baseline** for the local `football-master` environment in this repository.

The implementation is designed as a practical starting point for your project:

- it imports the local `football-master` source tree automatically
- it trains a **shared policy** for all controlled left-team players
- it works well with `simple115v2` observations for normal `11_vs_11_*` scenarios
- it keeps all custom code inside `Y_Fu` without modifying the upstream environment

## Current Focus

The current final target is:

- `five_vs_five`

Current practical priority order:

1. stabilize and evaluate `five_vs_five`
2. use `academy_*` only as curriculum or diagnostics
3. treat `11v11` as lower priority unless extra time remains

## Current Summary

The current `Y_Fu` line changed in an important way on `2026-04-02`.

The main takeaways are:

- the long `five_vs_five` PPO line still did not become good football
- reward-only revision and `player_id` both improved some behavior diagnostics, but still did not produce reliable match outcomes
- a no-ball action-validity bug was identified and fixed:
  - non-ball players no longer execute `pass` / `shot` / `dribble`
  - PPO now stores executed actions rather than only sampled actions
- behavior logging was extended with more football-process metrics
- the current roadmap therefore pauses blind `five_vs_five` continuation and returns to an Academy reboot with cleaner action semantics

If you want the shortest explanation of this pivot, start with:

- [PPO_POSTMORTEM.md](/home/yuhan/Codes/RL/Learning-to-Play-Football-Multi-Agent-Reinforcement-Learning/Y_Fu/PPO_POSTMORTEM.md)
- [ACADEMY_REBOOT_PLAN.md](/home/yuhan/Codes/RL/Learning-to-Play-Football-Multi-Agent-Reinforcement-Learning/Y_Fu/ACADEMY_REBOOT_PLAN.md)
- [ACADEMY_REBOOT_PAUSE_NOTE_2026-04-02.md](/home/yuhan/Codes/RL/Learning-to-Play-Football-Multi-Agent-Reinforcement-Learning/Y_Fu/ACADEMY_REBOOT_PAUSE_NOTE_2026-04-02.md)

## Next Plan

The immediate next plan is:

1. resume the Academy reboot from the saved `academy_pass_and_shoot_with_keeper` checkpoint
2. verify that invalid no-ball pass / shot / dribble behavior stays low under the new action filter
3. use the reboot run to judge whether `player_id` plus valid-action cleanup actually improves real pass-to-shot behavior
4. only after that, decide between:
   - a fresh `five_vs_five_reward_v2` ablation, or
   - another Academy stage handoff into `five_vs_five`

The practical rule is:

- do not spend another long `five_vs_five` PPO budget until the Academy reboot looks behaviorally cleaner

## Document Map

Use this section as the main index for the markdown files under `Y_Fu/`.

### Read First

- [README.md](/home/yuhan/Codes/RL/Learning-to-Play-Football-Multi-Agent-Reinforcement-Learning/Y_Fu/README.md)
  - main entry point for the `Y_Fu` folder
- [TRAINING_READING_GUIDE.md](/home/yuhan/Codes/RL/Learning-to-Play-Football-Multi-Agent-Reinforcement-Learning/Y_Fu/TRAINING_READING_GUIDE.md)
  - quickest way to reconstruct the current training line
- [TRAINING_STAGE_LOG.md](/home/yuhan/Codes/RL/Learning-to-Play-Football-Multi-Agent-Reinforcement-Learning/Y_Fu/TRAINING_STAGE_LOG.md)
  - current run log with `run completed` vs `stage passed`

### Current Training Execution

- [THREE_DAY_5V5_TRAINING_PLAN.md](/home/yuhan/Codes/RL/Learning-to-Play-Football-Multi-Agent-Reinforcement-Learning/Y_Fu/THREE_DAY_5V5_TRAINING_PLAN.md)
  - current hybrid `academy PPO -> 5v5 PPO -> 5v5 offline RL` schedule and resource plan
- [ACADEMY_TO_5V5_BOOTSTRAP_PLAN.md](/home/yuhan/Codes/RL/Learning-to-Play-Football-Multi-Agent-Reinforcement-Learning/Y_Fu/ACADEMY_TO_5V5_BOOTSTRAP_PLAN.md)
  - why Academy should be used as a primitive-learning bootstrap before `5v5`
- [ACADEMY_REBOOT_PLAN.md](/home/yuhan/Codes/RL/Learning-to-Play-Football-Multi-Agent-Reinforcement-Learning/Y_Fu/ACADEMY_REBOOT_PLAN.md)
  - current pivot: pause the failing `five_vs_five` PPO line and return to controlled Academy diagnosis
- [ACADEMY_REBOOT_PAUSE_NOTE_2026-04-02.md](/home/yuhan/Codes/RL/Learning-to-Play-Football-Multi-Agent-Reinforcement-Learning/Y_Fu/ACADEMY_REBOOT_PAUSE_NOTE_2026-04-02.md)
  - current pause status, completed simple tests, and the exact resume command for the Academy reboot
- [FIVE_V_FIVE_HALF_DAY_CHECKLIST.md](/home/yuhan/Codes/RL/Learning-to-Play-Football-Multi-Agent-Reinforcement-Learning/Y_Fu/FIVE_V_FIVE_HALF_DAY_CHECKLIST.md)
  - `5M / 10M / 20M` checkpoint criteria for the current `five_vs_five` run
- [FIVE_V_FIVE_EXECUTION_CHECKLIST.md](/home/yuhan/Codes/RL/Learning-to-Play-Football-Multi-Agent-Reinforcement-Learning/Y_Fu/FIVE_V_FIVE_EXECUTION_CHECKLIST.md)
  - concrete execution order, stop gates, and CPU-aware workflow for fixing the current `five_vs_five` failure
- [TRAINING_STAGE_LOG.md](/home/yuhan/Codes/RL/Learning-to-Play-Football-Multi-Agent-Reinforcement-Learning/Y_Fu/TRAINING_STAGE_LOG.md)
  - precise log of stage timing, evaluation, and representative videos

### Reward Design And PPO Reasoning

- [PPO_POSTMORTEM.md](/home/yuhan/Codes/RL/Learning-to-Play-Football-Multi-Agent-Reinforcement-Learning/Y_Fu/PPO_POSTMORTEM.md)
  - postmortem of the current PPO line, plus a football-like reward design direction that is harder to game
- [FIVE_V_FIVE_REWARD_V2_PLAN.md](/home/yuhan/Codes/RL/Learning-to-Play-Football-Multi-Agent-Reinforcement-Learning/Y_Fu/FIVE_V_FIVE_REWARD_V2_PLAN.md)
  - concrete next `five_vs_five` reward preset centered on attack quality and transition discipline
- [FIVE_V_FIVE_REWARD_V2_ABLATION_TABLE.md](/home/yuhan/Codes/RL/Learning-to-Play-Football-Multi-Agent-Reinforcement-Learning/Y_Fu/FIVE_V_FIVE_REWARD_V2_ABLATION_TABLE.md)
  - small controlled ablation across attack-quality, transition-discipline, and progression-focused reward variants
- [FIVE_V_FIVE_REWARD_ABLATION_RUNBOOK.md](/home/yuhan/Codes/RL/Learning-to-Play-Football-Multi-Agent-Reinforcement-Learning/Y_Fu/FIVE_V_FIVE_REWARD_ABLATION_RUNBOOK.md)
  - operating sequence, stop rules, and a logging template for running the reward-v2 ablation cleanly
- [REWARD_SHAPING_KEY_IDEA.md](/home/yuhan/Codes/RL/Learning-to-Play-Football-Multi-Agent-Reinforcement-Learning/Y_Fu/REWARD_SHAPING_KEY_IDEA.md)
  - explains how reward shaping affects PPO through `return -> advantage -> actor/critic update`
- [REWARD_REVISION_PROPOSAL.md](/home/yuhan/Codes/RL/Learning-to-Play-Football-Multi-Agent-Reinforcement-Learning/Y_Fu/REWARD_REVISION_PROPOSAL.md)
  - concrete proposal for revising the current shaping coefficients
- [FIVE_V_FIVE_FAILURE_SOLUTION_SWEEP.md](/home/yuhan/Codes/RL/Learning-to-Play-Football-Multi-Agent-Reinforcement-Learning/Y_Fu/FIVE_V_FIVE_FAILURE_SOLUTION_SWEEP.md)
  - diagnosis of the "10M steps and still garbage" failure pattern, plus a prioritized sweep of possible fixes
- [BEHAVIOR_DIAGNOSTICS_PLAN.md](/home/yuhan/Codes/RL/Learning-to-Play-Football-Multi-Agent-Reinforcement-Learning/Y_Fu/BEHAVIOR_DIAGNOSTICS_PLAN.md)
  - why behavior-level metrics like `pass_rate` and `right_bias` are now tracked for degenerate-policy detection
- [NO_BALL_ACTION_FILTER_PLAN.md](/home/yuhan/Codes/RL/Learning-to-Play-Football-Multi-Agent-Reinforcement-Learning/Y_Fu/NO_BALL_ACTION_FILTER_PLAN.md)
  - why no-ball `pass` / `shot` / `dribble` actions are now filtered and how PPO is aligned to executed actions
- [TRAINING_FAILURE_ARCHIVE.md](/home/yuhan/Codes/RL/Learning-to-Play-Football-Multi-Agent-Reinforcement-Learning/Y_Fu/TRAINING_FAILURE_ARCHIVE.md)
  - compact archive of past failure cases and the rule learned from each one

### Method Roadmap

- [TRAINING_METHOD_ROADMAP.md](/home/yuhan/Codes/RL/Learning-to-Play-Football-Multi-Agent-Reinforcement-Learning/Y_Fu/TRAINING_METHOD_ROADMAP.md)
  - two-layer roadmap: low-cost methods first, more systematic methods later
- [MARL_ROADMAP_SPEC.md](/home/yuhan/Codes/RL/Learning-to-Play-Football-Multi-Agent-Reinforcement-Learning/Y_Fu/MARL_ROADMAP_SPEC.md)
  - broader multi-agent stage roadmap
- [ACADEMY_TO_5V5_DECISION_TREE.md](/home/yuhan/Codes/RL/Learning-to-Play-Football-Multi-Agent-Reinforcement-Learning/Y_Fu/ACADEMY_TO_5V5_DECISION_TREE.md)
  - post-PPO decision tree and a breakdown of what Academy can and cannot realistically teach
- [RESEARCH_TODO_SPEC.md](/home/yuhan/Codes/RL/Learning-to-Play-Football-Multi-Agent-Reinforcement-Learning/Y_Fu/RESEARCH_TODO_SPEC.md)
  - how outside papers should be used to improve the `Y_Fu` line

### Offline RL

- [PPO_OFFLINE_RL_INTEGRATION_PLAN.md](/home/yuhan/Codes/RL/Learning-to-Play-Football-Multi-Agent-Reinforcement-Learning/Y_Fu/PPO_OFFLINE_RL_INTEGRATION_PLAN.md)
  - recommended way to combine Academy PPO, `5v5` PPO, IQL, and PPO fine-tuning
- [OFFLINE_RL_ROADMAP.md](/home/yuhan/Codes/RL/Learning-to-Play-Football-Multi-Agent-Reinforcement-Learning/Y_Fu/OFFLINE_RL_ROADMAP.md)
  - full offline RL roadmap, dataset plan, IQL design, and iteration loop
- [OFFLINE_RL_EXECUTION_PLAN.md](/home/yuhan/Codes/RL/Learning-to-Play-Football-Multi-Agent-Reinforcement-Learning/Y_Fu/OFFLINE_RL_EXECUTION_PLAN.md)
  - operational staged plan for using offline RL on `5v5` after PPO transfer
- [OFFLINE_RL_COMMANDS.md](/home/yuhan/Codes/RL/Learning-to-Play-Football-Multi-Agent-Reinforcement-Learning/Y_Fu/OFFLINE_RL_COMMANDS.md)
  - ready-to-run commands for the hybrid `academy PPO -> 5v5 PPO -> IQL -> PPO` loop
- [PPO_IQL_EXECUTION_CHECKLIST.md](/home/yuhan/Codes/RL/Learning-to-Play-Football-Multi-Agent-Reinforcement-Learning/Y_Fu/PPO_IQL_EXECUTION_CHECKLIST.md)
  - short execution checklist for the hybrid PPO and IQL workflow

### Reports And Results

- [multiagent_eval_report.md](/home/yuhan/Codes/RL/Learning-to-Play-Football-Multi-Agent-Reinforcement-Learning/Y_Fu/reports/multiagent_eval_report.md)
  - current evaluation report for the multi-agent line

### Legacy / Supplemental Notes

- [SPEC.md](/home/yuhan/Codes/RL/Learning-to-Play-Football-Multi-Agent-Reinforcement-Learning/Y_Fu/SPEC.md)
  - unified current high-level spec for the `Y_Fu` line
- [TERMINAL_COMMANDS.md](/home/yuhan/Codes/RL/Learning-to-Play-Football-Multi-Agent-Reinforcement-Learning/Y_Fu/TERMINAL_COMMANDS.md)
  - synchronized command reference for the current hybrid workflow
- [SALTYFISH_UPGRADE_SPEC.md](/home/yuhan/Codes/RL/Learning-to-Play-Football-Multi-Agent-Reinforcement-Learning/Y_Fu/SALTYFISH_UPGRADE_SPEC.md)
  - notes for the separate `saltyfish_baseline` line

### Which Files Matter Most Right Now

If you only want the current active line, read these in order:

1. [SPEC.md](/home/yuhan/Codes/RL/Learning-to-Play-Football-Multi-Agent-Reinforcement-Learning/Y_Fu/SPEC.md)
2. [TRAINING_STAGE_LOG.md](/home/yuhan/Codes/RL/Learning-to-Play-Football-Multi-Agent-Reinforcement-Learning/Y_Fu/TRAINING_STAGE_LOG.md)
3. [PPO_POSTMORTEM.md](/home/yuhan/Codes/RL/Learning-to-Play-Football-Multi-Agent-Reinforcement-Learning/Y_Fu/PPO_POSTMORTEM.md)
4. [ACADEMY_REBOOT_PLAN.md](/home/yuhan/Codes/RL/Learning-to-Play-Football-Multi-Agent-Reinforcement-Learning/Y_Fu/ACADEMY_REBOOT_PLAN.md)
5. [ACADEMY_REBOOT_PAUSE_NOTE_2026-04-02.md](/home/yuhan/Codes/RL/Learning-to-Play-Football-Multi-Agent-Reinforcement-Learning/Y_Fu/ACADEMY_REBOOT_PAUSE_NOTE_2026-04-02.md)
6. [NO_BALL_ACTION_FILTER_PLAN.md](/home/yuhan/Codes/RL/Learning-to-Play-Football-Multi-Agent-Reinforcement-Learning/Y_Fu/NO_BALL_ACTION_FILTER_PLAN.md)
7. [FIVE_V_FIVE_REWARD_V2_PLAN.md](/home/yuhan/Codes/RL/Learning-to-Play-Football-Multi-Agent-Reinforcement-Learning/Y_Fu/FIVE_V_FIVE_REWARD_V2_PLAN.md)
8. [TRAINING_FAILURE_ARCHIVE.md](/home/yuhan/Codes/RL/Learning-to-Play-Football-Multi-Agent-Reinforcement-Learning/Y_Fu/TRAINING_FAILURE_ARCHIVE.md)

## What Is Implemented

- `train.py`: trains a PPO agent
- `train_lightning.py`: launches a very fast starter run
- `evaluate.py`: loads a saved checkpoint and runs evaluation episodes
- `yfu_football/envs.py`: environment wrapper for Google Research Football
- `yfu_football/model.py`: actor-critic network
- `yfu_football/ppo.py`: rollout collection, GAE, PPO updates, checkpointing

## Available Models

- `cnn`: best for `extracted` or other image-like observations
- `residual_mlp`: stronger default for `simple115v2`
- `separate_mlp`: separate actor and critic backbones for larger runs
- `mlp`: plain baseline
- `auto`: chooses `cnn` for image-like observations and `residual_mlp` for vector observations

## Recommended Progression

Do not start from full `11_vs_11` immediately.

Recommended order:

1. `academy_run_to_score_with_keeper`
2. `academy_pass_and_shoot_with_keeper`
3. `academy_3_vs_1_with_keeper`
4. `five_vs_five`
5. `small_11v11`
6. `full_11v11_residual`

These stages move from short attacking drills to small-team football and only then to full matches.

## Default Training Setup

The default command trains on:

- scenario: `11_vs_11_easy_stochastic`
- observation: `simple115v2`
- reward: `scoring,checkpoints`
- controlled players: `11`

That means this baseline is already aligned with the multi-agent direction of the project, but it uses **parameter-sharing PPO** rather than a centralized-critic MAPPO implementation.

## Before Running

You need a working Google Research Football build in `football-master/`.

Activate the shared environment from the repository root before running anything in `Y_Fu/`:

```bash
source football-master/football-env/bin/activate
```

The commands below assume that `gfootball` is already installed and that PyTorch is available in `football-env`.

## Train

From the repository root:

```bash
source football-master/football-env/bin/activate
python Y_Fu/train.py --device cpu
```

For the fastest first run, use the lightning preset:

```bash
source football-master/football-env/bin/activate
python Y_Fu/train_lightning.py --device cpu
```

This quick-start preset uses:

- scenario: `academy_empty_goal_close`
- observation: `extracted`
- controlled players: `1`
- timesteps: `20000`
- smaller network: `128 128`

The same run is also available from the main trainer:

```bash
python Y_Fu/train.py --preset lightning --device cpu
```

Recommended first real training stage:

```bash
python Y_Fu/train.py --preset academy_run_to_score_with_keeper --device cpu
```

Passing-focused stage:

```bash
python Y_Fu/train.py --preset academy_pass_and_shoot_with_keeper --device cpu
```

Small attacking group stage:

```bash
python Y_Fu/train.py --preset academy_3_vs_1_with_keeper --device cpu
```

Small-team football stage:

```bash
python Y_Fu/train.py --preset five_vs_five --device cpu
```

Example with a smaller custom quick test run:

```bash
python Y_Fu/train.py --total-timesteps 20000 --rollout-steps 128 --save-interval 2 --device cpu
```

Example keeping the football match setting but shrinking it:

```bash
python Y_Fu/train.py --preset small_11v11 --device cpu
```

Wider 3-player run with separate actor and critic:

```bash
python Y_Fu/train.py --preset small_11v11_wide --device cpu
```

Recommended stronger 11v11 run:

```bash
python Y_Fu/train.py --preset full_11v11_residual --device cpu
```

Alternative larger 11v11 run with separate actor and critic:

```bash
python Y_Fu/train.py --preset full_11v11_wide --device cpu
```

## Evaluate

```bash
source football-master/football-env/bin/activate
python Y_Fu/evaluate.py --checkpoint Y_Fu/checkpoints/latest.pt --episodes 3 --deterministic --device cpu
```

Compare your checkpoint against a random-action benchmark:

```bash
python Y_Fu/evaluate.py --checkpoint Y_Fu/checkpoints/latest.pt --episodes 5 --deterministic --compare-random --device cpu
```

Watch a rendered evaluation episode:

```bash
python Y_Fu/evaluate.py --checkpoint Y_Fu/checkpoints/latest.pt --episodes 1 --deterministic --render --device cpu
```

Benchmark metrics reported:

- `avg_return`
- `avg_score_reward`
- `avg_goal_diff`
- `win_rate`
- `avg_length`

## Notes

- In the current setup, `--device cpu` is the safe default.
- Keep one shared `train.py` and switch experiments by preset instead of creating many separate trainer files.
- `simple115v2` is intended for normal-game scenarios, especially `11_vs_11_*`.
- The academy presets and `five_vs_five` use `extracted` observations with the `cnn` model.
- The `lightning` preset is only a quick sanity check. It uses a 1-player academy scenario, not a full football match.
- If a checkpoint performs badly in 11v11, prefer `residual_mlp` or `separate_mlp` over the plain MLP baseline.
- Episode return in the logs is the **mean reward across controlled players** over the episode.
- `score_reward`, `goal_diff`, and `win_rate` are useful to compare your policy against the random baseline.
- If you later want true MAPPO, this code is a good baseline to extend by replacing the value function with a centralized critic.
