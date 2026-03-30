# Y_Fu Spec

## Purpose

This file is the working spec for the `Y_Fu/` training code.

If a future session continues this work, start from this file first.

For the stage-by-stage multi-agent roadmap and completion criteria, also read:

- `Y_Fu/MARL_ROADMAP_SPEC.md`

## Naive Phase

Everything up to this point should be treated as a naive exploratory phase rather than a final method.

Characteristics of this phase:

- the main goal was to get the environment working and obtain nontrivial behavior as quickly as possible
- training choices were driven by practical iteration rather than a careful method study
- reward shaping and curriculum steps were added incrementally based on observed failures
- checkpoint selection was used pragmatically when `latest.pt` was not the best model
- video inspection was used heavily to diagnose behavior

Limits of this phase:

- the football environment has not yet been studied systematically
- the design has not yet been aligned carefully with methods from the course
- the current approach is still mostly shared-policy PPO with hand-added shaping
- the current results should be treated as rough baselines and engineering probes, not final conclusions

Main purpose of the naive phase:

- verify the environment and code path
- establish a working curriculum
- identify the real bottlenecks
- gather evidence about what fails first:
  - sparse reward
  - poor attack
  - weak cooperation
  - weak transition defense

The next stage should be more deliberate:

- study the environment and scenario structure more carefully
- connect the design to methods learned in the course
- decide which algorithmic changes are principled rather than only heuristic
- use the current checkpoints, videos, and logs as baseline reference points

## Paper Framing

For the paper, the main research question is multi-agent cooperation, coordination, and RL methods in football.

That means:

- treat Kaggle and the SaltyFish-style single-player line as engineering baselines or side references
- do not treat Kaggle competition writeups as the main conceptual reference for the paper

Better main references for the paper are:

- MAPPO / centralized-critic MARL papers
- cooperative multi-agent control papers
- football or soccer MARL papers where several agents are learned together

Research implication:

- the multi-player `Y_Fu/train.py` branch should be the main paper direction
- the single-player `Y_Fu/train_saltyfish.py` branch should remain a baseline or comparison line

## Tomorrow Start Here

The single-player SaltyFish line should be stopped as a main research direction.

Reason:

- it is only a single-player control setup inside full `11v11`
- after a long run, it still did not become a strong enough result to justify more main effort
- the paper question is multi-agent cooperation, so the main line should move back to the multi-player branch

Tomorrow's main action:

1. Do not continue `Y_Fu/train_saltyfish.py` as the main line.
2. Start the 2-agent stage:

```bash
cd ~/Codes/RL/Learning-to-Play-Football-Multi-Agent-Reinforcement-Learning
source football-master/football-env/bin/activate
python -u Y_Fu/train.py --preset academy_pass_and_shoot_with_keeper --device cpu
```

3. After training, evaluate with:

```bash
python Y_Fu/evaluate.py --checkpoint Y_Fu/checkpoints/academy_pass_and_shoot_with_keeper/latest.pt --episodes 20 --deterministic --compare-random --device cpu
```

4. Also evaluate at least one earlier nearby checkpoint, not only `latest.pt`.
5. Use `Y_Fu/MARL_ROADMAP_SPEC.md` as the criterion for whether the 2-agent stage is finished.

Advance rule for tomorrow's stage:

- advance to the 3-agent stage only if the 2-agent stage clearly beats random and video shows real pass-to-shot cooperation

If the 2-agent stage is unstable:

- keep working on the 2-agent stage before moving up the curriculum

## Next Multi-Agent Experiment

The next main experiment should move back to the multi-player `Y_Fu/train.py` branch and treat the single-player SaltyFish line as a baseline only.

### Main Question

How do several learned players coordinate in football under shared-policy PPO and reward shaping?

Main behaviors of interest:

- passing to teammates under pressure
- retaining possession instead of forcing low-value dribbles
- entering the final third with support
- recovering shape after losing the ball
- transition defense and possession recovery

### Recommended Next Stage

Use `five_vs_five` as the main next experiment.

Reason:

- `academy_*` stages are useful for building attacking primitives
- `five_vs_five` is the first stage where cooperation and transition behavior become meaningful
- it is much more aligned with the paper question than the single-player `11_vs_11_kaggle` line

### Controlled Players

For the current `five_vs_five` preset in code:

- environment: `5_vs_5`
- controlled players: `4`
- observation: `extracted`
- model: `cnn`
- algorithm: shared-policy PPO

This is already a genuine multi-player learning setup.

### Exact Next Command

Run this as the next main training line:

```bash
cd ~/Codes/RL/Learning-to-Play-Football-Multi-Agent-Reinforcement-Learning
source football-master/football-env/bin/activate
python -u Y_Fu/train.py --preset five_vs_five --device cpu --rollout-steps 1024 --total-timesteps 2000000 --init-checkpoint Y_Fu/checkpoints/five_vs_five/update_140.pt --pass-success-reward 0.08 --pass-failure-penalty 0.03 --pass-progress-reward-scale 0.08 --shot-attempt-reward 0.03 --attacking-possession-reward 0.0015 --final-third-entry-reward 0.04 --possession-retention-reward 0.0010 --own-half-turnover-penalty 0.015 --possession-recovery-reward 0.02 --defensive-third-recovery-reward 0.02 --opponent-attacking-possession-penalty 0.0005
```

This should be treated as the main paper training command until a stronger MARL algorithm replaces shared-policy PPO.

### Main Metrics For The Paper

Top-level outcome metrics:

- `avg_goal_diff`
- `win_rate`
- `avg_score_reward`

Cooperation-oriented metrics to report:

- pass success rate
- number of completed passes per episode
- final-third entry frequency
- possession retention after pass reception
- possession recovery frequency
- own-half turnover frequency
- goals for and goals against

Interpretation rule:

- goals and win rate are the final outcome
- cooperation metrics are the intermediate evidence that coordination is actually improving

### Comparison Structure

The paper should compare at least:

1. Random baseline
2. Earlier weak shared-policy PPO baseline
3. Reward-shaped multi-player PPO in `five_vs_five`
4. If time allows, a stronger MARL variant such as centralized-critic PPO / MAPPO-style training

### Near-Term Algorithm Direction

After stabilizing the `five_vs_five` line, the next algorithmically meaningful step is:

- add a centralized critic while keeping decentralized execution

That would align the project much better with MAPPO-style MARL literature than the current single-player Kaggle-style baseline.

## SaltyFish-Inspired Baseline

A separate baseline has been added under `Y_Fu/saltyfish_baseline/` to capture the parts of the public SaltyFish summary that are practical to reproduce locally.

Detailed follow-up diagnosis and the upgraded local baseline plan are recorded in:

- `Y_Fu/SALTYFISH_UPGRADE_SPEC.md`

What this baseline does:

- uses the competition-style single-player scenario `11_vs_11_kaggle`
- uses `simple115v2` observations
- groups the simple115 features into separate semantic heads before merging them
- uses a reduced static action set
- trains with plain local PPO on one machine

What it does not reproduce exactly:

- distributed IMPALA training
- full self-play league training
- behavior cloning from competition trajectories
- Kaggle ladder evaluation setup

Purpose:

- provide a cleaner competition-style baseline than the rough multi-player shaping experiments
- give a more systematic reference point before designing a course-informed method

Main entrypoints:

- `python Y_Fu/train_saltyfish.py --device cpu`
- `python Y_Fu/evaluate_saltyfish.py --checkpoint Y_Fu/checkpoints/saltyfish_baseline/latest.pt --episodes 5 --compare-random --device cpu`

## Environment

- Repository root:
  - `~/Codes/RL/Learning-to-Play-Football-Multi-Agent-Reinforcement-Learning`
- Shared environment:
  - `football-master/football-env`
- Activate from repository root:

```bash
source football-master/football-env/bin/activate
```

## Main Entry Points

- Training:
  - `python Y_Fu/train.py ...`
- Lightning sanity check:
  - `python Y_Fu/train_lightning.py --device cpu`
- Evaluation:
  - `python Y_Fu/evaluate.py ...`

## Current Design

- Keep one shared trainer implementation in `Y_Fu/yfu_football/ppo.py`
- Keep one evaluation script in `Y_Fu/evaluate.py`
- Switch experiments by preset, not by creating many separate trainer files

## Upgrade Roadmap

The project has been upgraded in this order:

1. Get the shared `football-master/football-env` working and verify basic `gfootball` import and environment creation.
2. Add a stable PPO training path in `Y_Fu` with checkpoint saving and evaluation.
3. Add a curriculum instead of jumping straight to full football:
   - `academy_run_to_score_with_keeper`
   - `academy_pass_and_shoot_with_keeper`
   - `academy_3_vs_1_with_keeper`
   - `five_vs_five`
4. Add a model zoo and preset system so different experiments can share one trainer.
5. Improve logging:
   - `episodes_finished`
   - `score_reward`
   - `episode_length_range`
   - `success_rate`
   - `goals_for`
   - `goals_against`
   - `score_examples`
6. Add seeded evaluation and video export in `Y_Fu/evaluate.py`.
7. Realize that `latest.pt` is not always the best checkpoint and start selecting earlier checkpoints by evaluation.
8. Find that `academy_3_vs_1_with_keeper/update_90.pt` was a better attacking checkpoint than `latest.pt`.
9. Transfer from `academy_3_vs_1_with_keeper/update_90.pt` into `five_vs_five`.
10. Increase `five_vs_five` rollout length from `512` to `1024` for longer match context per PPO update.
11. Add offense-focused reward shaping:
   - pass success reward
   - pass failure penalty
   - shot attempt reward
   - attacking possession reward
12. Add a second shaping pass:
   - pass progress reward
   - final-third entry reward
   - possession retention reward
   - own-half turnover penalty
13. Video diagnosis after the offense-focused run:
   - the agent became more offensive
   - but strategic cooperation was still weak
   - and defensive transition was too slow or ineffective
14. Add a balancing defensive shaping pass:
   - possession recovery reward
   - defensive-third recovery reward
   - opponent attacking possession penalty
15. Observe that the first offense+defense shaping mix was too punitive:
   - `goals_for` still stayed at zero
   - several scorelines remained `0-x`
   - even some `0-0` draws had negative `episode_return`
16. Prepare a softer restart with lower penalties while keeping the positive offense signals.

Current intended next training command after the last upgrade:

```bash
python -u Y_Fu/train.py --preset five_vs_five --device cpu --rollout-steps 1024 --total-timesteps 2000000 --init-checkpoint Y_Fu/checkpoints/five_vs_five/update_140.pt --pass-success-reward 0.08 --pass-failure-penalty 0.03 --pass-progress-reward-scale 0.08 --shot-attempt-reward 0.03 --attacking-possession-reward 0.0015 --final-third-entry-reward 0.04 --possession-retention-reward 0.0010 --own-half-turnover-penalty 0.015 --possession-recovery-reward 0.02 --defensive-third-recovery-reward 0.02 --opponent-attacking-possession-penalty 0.0005
```

## Current Milestone

Latest evaluated `five_vs_five` result before the new shaping rerun:

- trained policy:
  - `avg_goal_diff = -1.900`
  - `avg_score_reward = -1.900`
  - `win_rate = 0.000`
- random policy:
  - `avg_goal_diff = -2.800`
  - `avg_score_reward = -2.800`
  - `win_rate = 0.000`
- delta vs random:
  - `avg_goal_diff = +0.900`

Interpretation:

- this is better than the older `five_vs_five` runs
- the policy is clearly better than random
- but it still cannot score reliably and still loses most matches
- the main remaining bottleneck is offense, not basic survival
- later video inspection showed that once offense improved, the next bottlenecks became strategic cooperation and transition defense
- the first stronger defense-penalty mix appears to have been too harsh, so the next restart should reduce the penalties and preserve the attacking incentives

## Reproducible Video Capture

If an interesting evaluation episode appears in an unseeded run, the log proves it happened, but that exact historical episode is not guaranteed to be recoverable later. To make video capture reproducible, rerun evaluation with a fixed seed and save video.

Example:

```bash
python Y_Fu/evaluate.py --checkpoint Y_Fu/checkpoints/five_vs_five/latest.pt --episodes 5 --deterministic --device cpu --seed 123 --save-video --video-dir Y_Fu/videos/five_vs_five_seed123
```

With the same checkpoint and the same `--seed`, the episode sequence should repeat, so if episode 4 is `0-0` in that seeded run, you can regenerate the same video again later.

## What A Checkpoint Is

A checkpoint is a saved snapshot of training state at a particular update. In this project, a checkpoint stores:

- model weights
- optimizer state
- training config
- observation/action dimensions
- update number
- total agent steps

So a checkpoint lets you:

- evaluate a past version of the policy
- continue training from that saved state
- compare earlier and later policies

`latest.pt` is only the most recently saved checkpoint in that preset folder. It is not guaranteed to be the best-performing checkpoint. Earlier checkpoints such as `update_90.pt` can be better if the policy later drifts into a weaker or more conservative behavior.

## Available Model Types

- `cnn`
- `residual_mlp`
- `separate_mlp`
- `mlp`
- `auto`

## Recommended Curriculum

Run these in order:

1. `academy_run_to_score_with_keeper`
2. `academy_pass_and_shoot_with_keeper`
3. `academy_3_vs_1_with_keeper`
4. `five_vs_five`
5. `small_11v11`
6. `full_11v11_residual`

## Current Presets

- `lightning`
- `academy_run_to_score_with_keeper`
- `academy_pass_and_shoot_with_keeper`
- `academy_3_vs_1_with_keeper`
- `five_vs_five`
- `small_11v11`
- `small_11v11_wide`
- `full_11v11_residual`
- `full_11v11_wide`

## Recommended Commands

### Train

```bash
python Y_Fu/train.py --preset academy_run_to_score_with_keeper --device cpu
python Y_Fu/train.py --preset academy_pass_and_shoot_with_keeper --device cpu
python Y_Fu/train.py --preset academy_3_vs_1_with_keeper --device cpu
python Y_Fu/train.py --preset five_vs_five --device cpu
python Y_Fu/train.py --preset small_11v11 --device cpu
python Y_Fu/train.py --preset full_11v11_residual --device cpu
```

### Evaluate

```bash
python Y_Fu/evaluate.py --checkpoint Y_Fu/checkpoints/latest.pt --episodes 3 --deterministic --device cpu
python Y_Fu/evaluate.py --checkpoint Y_Fu/checkpoints/latest.pt --episodes 5 --deterministic --compare-random --device cpu
python Y_Fu/evaluate.py --checkpoint Y_Fu/checkpoints/latest.pt --episodes 1 --deterministic --render --device cpu
python Y_Fu/evaluate.py --checkpoint Y_Fu/checkpoints/latest.pt --episodes 1 --deterministic --device cpu --save-video --video-dir Y_Fu/videos
```

For preset-specific checkpoints, replace `Y_Fu/checkpoints/latest.pt` with the corresponding path, for example:

```bash
Y_Fu/checkpoints/full_11v11_residual/latest.pt
Y_Fu/checkpoints/five_vs_five/latest.pt
```

## Current Diagnosis

- The old full `11_vs_11` baseline is structurally weak.
- The agent showed bad behavior in video:
  - goalkeeper leaves position
  - ball carrier dribbles badly instead of passing
- Main reasons:
  - one shared policy for all roles
  - no role-specific reward
  - checkpoint reward biases forward dribbling
  - no centralized critic
  - full `11_vs_11` is too hard as a starting point

## Logging

The PPO trainer now prints:

- `episodes_finished`
- `episode_return`
- `score_reward`
- `episode_length`
- `episode_length_range`
- `success_rate`

`success_rate` is computed per completed episode as:

- win if final score is available and left score > right score
- otherwise success if cumulative `score_reward > 0`

`episodes_finished` means how many episodes reached a terminal state during the current fixed rollout window. PPO collects a fixed chunk of steps per update, so an episode can continue across update boundaries and finish in a later update.

For academy scenarios such as `academy_3_vs_1_with_keeper`, an episode usually ends when:

- a goal is scored
- the defending side gains possession
- the ball goes out of play

This means:

- short episode + positive `score_reward` usually means quick success
- short episode + zero `score_reward` usually means turnover or ball out
- positive `episode_return` with zero `score_reward` means shaping reward was collected, but no actual goal was scored

For `five_vs_five`, episodes behave more like full matches. In practice, an episode usually ends at the scenario's full game terminal condition, often the built-in match length limit. That is why `episode_length` can be around `3001` and why many updates can show `episodes_finished=0`.

This means:

- academy tasks have short, frequent terminal events
- `five_vs_five` has long episodes and much sparser completed-episode feedback

`rollout_steps` is the fixed number of environment steps collected per PPO update. If the rollout window is short relative to episode length, that is not an error by itself. It simply means:

- many updates will contain no completed episodes
- training logs will often show `episode_return=n/a`
- reward and value learning become noisier because terminal feedback is seen less often

It only becomes a practical problem when the rollout is so short that learning is too slow or unstable for the long-horizon task.

Example for `five_vs_five`:

- one full match can last about `3001` steps
- one PPO update with `rollout_steps=512` collects `512` environment steps

So one full match spans about `3001 / 512 ~= 5.86` rollout windows, which is roughly 6 updates. That is why:

- many consecutive updates can show `episodes_finished=0`
- then one later update shows `episodes_finished=1`

This is normal. The episode is continuing across several PPO updates and only finishes when one update happens to include the terminal step.

## Value Function In Sparse Reward

In the current setup, the value function is learned from rollouts using:

- rewards collected along the episode
- bootstrapping with `V(next_state)`
- GAE smoothing

So even if the final goal happens late, the critic can still learn earlier estimates such as:

- this state tends to lead to success more often
- this state usually collapses into failure

That is the main purpose of the value function in sparse-reward RL.

There are still clear limits:

- if reward is very rare
- if episodes are long
- if the state/action space is large

then the value estimate becomes noisy and hard to learn. This is one of the main reasons full football is much harder than the academy curriculum stages.

MCTS is usually not the practical next step for this project. Football is harder for MCTS because:

- it is real-time and long-horizon
- it is multi-agent
- action branching compounds over time
- control feels continuous even though the action space is discrete
- online tree search would be expensive to run repeatedly

More practical ways to handle sparse reward here are:

- curriculum learning
- reward shaping
- better checkpoint selection
- larger training budget
- role-aware architectures
- centralized critic or MAPPO-style methods

## Pure RL Vs Guided Shaping

The reward shaping used in `Y_Fu` injects human knowledge. This is less pure than an AlphaZero-style approach and should be treated as a practical engineering choice, not as a claim of fully autonomous discovery.

Tradeoff:

- minimal human input is more elegant and more faithful to pure RL
- but it is much harder to train in this football setting
- guided shaping is less pure
- but more practical for a small project with limited compute

In practice for this project, some shaping is often necessary just to get traction. Without it, PPO may never discover useful attacking behavior at all.

## Important Checkpoint Note

- New presets save into separate folders so they do not overwrite each other.
- `Y_Fu/checkpoints/latest.pt` was restored to the older longer run and should not be treated as the best model automatically.
- Prefer evaluating the checkpoint inside the preset-specific folder you just trained.
