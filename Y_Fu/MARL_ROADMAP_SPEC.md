# Y_Fu Multi-Agent Roadmap Spec

## Purpose

This file defines the main multi-agent research roadmap for `Y_Fu/`.

Use it to decide:

- what the main training stage is
- when a stage is finished
- when to transfer to the next stage
- when to stop polishing one stage and move on

This roadmap is for the multi-player `Y_Fu/train.py` branch, not the single-player SaltyFish baseline.

## Intended Multi-Agent Formulation

Target formulation:

- each controlled player acts from its own observation at execution time
- agents may share one policy network through parameter sharing
- training may later use a centralized critic

So the intended direction is:

- decentralized execution
- parameter-shared actors
- centralized critic as a later upgrade target

This is the clean MARL direction for the paper.

### Important Distinction

What is desired conceptually:

- decentralized execution:
  - each player acts from its own local observation
- parameter sharing is allowed:
  - all players may still use the same policy network weights
  - they are still separate agents in execution
- centralized training may be added:
  - especially through a centralized critic

What is not the desired final formulation:

- one single global controller with full state directly commanding the whole team as a monolithic action source

Standard MARL interpretation:

- CTDE: centralized training, decentralized execution

Current status of the `Y_Fu/train.py` branch:

- already partly decentralized:
  - each controlled player gets its own observation
  - each controlled player produces its own action
- currently uses a shared policy without a centralized critic
- therefore it is a useful shared-policy PPO baseline, but not yet full MAPPO-style CTDE

## Stage Order

Recommended main order:

1. `academy_pass_and_shoot_with_keeper`
2. `academy_3_vs_1_with_keeper`
3. `five_vs_five`
4. `11_vs_11_easy_stochastic`

Current preset-controlled players:

- `academy_pass_and_shoot_with_keeper`: `2`
- `academy_3_vs_1_with_keeper`: `3`
- `five_vs_five`: `4`
- `full_11v11_residual`: `11`

## General Stage Rule

A stage is finished only if all three conditions hold:

1. Outcome condition:
   - clearly better than random or an earlier weak checkpoint
2. Behavior condition:
   - videos show the intended cooperative behavior for that stage
3. Stability condition:
   - improvement is visible across multiple checkpoints or repeated evaluations, not only one lucky run

If only the outcome improves but behavior is still wrong, the stage is not finished.

If behavior looks better but metrics are unstable, the stage is not finished.

## Stage 1: 2-Agent Stage

Preset:

- `academy_pass_and_shoot_with_keeper`

Research purpose:

- learn basic cooperative attack
- learn that passing is often better than forced solo dribbling

Main behaviors expected:

- use the second attacker instead of always carrying alone
- create a shot after a useful pass
- reduce bad pass failures and dead-end carries

Completion criteria:

- success rate is clearly above random
- pass success is frequent and repeatable in videos
- the policy regularly uses the free teammate
- shots are often created from cooperative movement rather than accidental loose-ball situations

Operational completion threshold:

- evaluate at least 20 deterministic episodes against random
- require the trained checkpoint to beat random on success / win behavior and score outcome
- practical threshold for "good enough to advance":
  - success / win behavior is clearly above random over 20 episodes
  - `avg_score_reward` is clearly above random
  - videos from at least 2 episodes show purposeful pass-to-shot cooperation
  - this should hold for at least 2 nearby checkpoints, not just one

Practical decision rule:

- advance:
  - if two nearby checkpoints both clearly beat random and the videos show real cooperation
- repeat:
  - if one checkpoint looks good but neighboring checkpoints are unstable
- redesign:
  - if long training still shows mostly solo-dribble behavior and no consistent advantage over random

Not finished if:

- the lead player still solves most episodes by solo dribble only
- passes happen but are random and not connected to shot creation
- success appears in logs but not in repeated evaluation

Transfer rule:

- once the stage reliably shows purposeful passing and clear success over random, move to Stage 2

Exact next training command for this stage:

```bash
cd ~/Codes/RL/Learning-to-Play-Football-Multi-Agent-Reinforcement-Learning
source football-master/football-env/bin/activate
python -u Y_Fu/train.py --preset academy_pass_and_shoot_with_keeper --device cpu
```

Exact evaluation command for this stage:

```bash
python Y_Fu/evaluate.py --checkpoint Y_Fu/checkpoints/academy_pass_and_shoot_with_keeper/latest.pt --episodes 20 --deterministic --compare-random --device cpu
```

Checkpoint-selection rule for this stage:

- evaluate `latest.pt` and at least one earlier checkpoint such as `update_10.pt`, `update_20.pt`, or the best recent saved checkpoint
- if `latest.pt` is weaker, transfer from the better earlier checkpoint instead

## Stage 2: 3-Agent Stage

Preset:

- `academy_3_vs_1_with_keeper`

Research purpose:

- move from pairwise cooperation to small-group attacking coordination

Main behaviors expected:

- choose between multiple teammates instead of one fixed passing option
- use the extra attacker to avoid pressure
- create better shooting lanes through support positioning

Completion criteria:

- clear and stable success advantage over random
- videos show real 3-player cooperation rather than repeated single-player carries
- the ball is not trapped by one attacker for most possessions
- earlier good checkpoints do not immediately collapse when evaluated repeatedly

Not finished if:

- success still depends mostly on one player dribbling through
- teammates are present but mostly ignored
- a good-looking checkpoint is not reproducible

Transfer rule:

- once 3-player attack is clearly purposeful and repeatable, transfer to `five_vs_five`

## Stage 3: 5v5 Stage

Preset:

- `five_vs_five`

Research purpose:

- first real small-team football stage
- study coordination plus transition defense

Main behaviors expected:

- coordinated possession instead of panic clearances or forced carries
- support after pass reception
- recovery after turnover
- some defensive shape and transition behavior
- final-third entries created by cooperation rather than chaos

Primary metrics:

- `avg_goal_diff`
- `win_rate`
- `avg_score_reward`

Cooperation metrics to add or track:

- pass success rate
- completed passes per episode
- final-third entry frequency
- possession retention after pass reception
- possession recovery frequency
- own-half turnover frequency
- goals for
- goals against

Completion criteria:

- clearly better than random on repeated evaluation
- lower goals against than earlier checkpoints or random
- visible team cooperation in video
- turnovers in own half decrease
- final-third entries and useful passes happen regularly

Not finished if:

- the team is only surviving without attacking
- goal-diff improvements are tiny and unstable
- passes happen but do not improve team attack or retention
- transition defense is still obviously broken

Transfer rule:

- move to 11v11 only after 5v5 shows stable improvement both in outcome and cooperation metrics

## Stage 4: 11v11 Stage

Preset:

- `full_11v11_residual`

Research purpose:

- large-scale validation of the learned multi-agent approach

Main behaviors expected:

- preserve 5v5 cooperation patterns at a larger scale
- maintain possession under more realistic spacing
- show broader team shape in attack and defense
- reduce goals conceded while still creating chances

Primary metrics:

- `avg_goal_diff`
- `win_rate`
- `avg_score_reward`
- goals for
- goals against

Secondary multi-agent metrics:

- pass success rate
- final-third entry frequency
- possession recovery frequency
- own-half turnover frequency
- draw rate

Completion criteria:

- clear improvement over random and earlier weak checkpoints
- reduced goals against relative to earlier 11v11 runs
- some nonzero scoring ability, not only survival
- videos show coordinated team behavior rather than isolated local improvements

Not finished if:

- the team still mostly survives without creating attack
- one checkpoint looks good but later evaluation does not support it
- large-scale spacing destroys the cooperative behaviors learned in smaller stages

## Decision Rules

### Advance

Advance to the next stage only when:

- repeated evaluation is clearly better than random
- the intended behavior for the current stage is visible in video
- the next stage is the natural next source of difficulty

### Repeat

Repeat the current stage when:

- behavior is promising but unstable
- metrics are improving but not yet convincingly
- the next stage would add too many difficulties at once

### Stop And Redesign

Stop and redesign the method when:

- long training still does not beat random
- video shows the same failure mode repeating
- reward shaping is clearly pushing the wrong behavior
- the stage has become a dead end instead of a stepping stone

## Current Recommended Main Line

At the moment, the main research line should be:

1. use earlier academy stages only as preparation
2. make `five_vs_five` the main current experiment
3. move to `11v11` only after 5v5 is stable
4. then consider a stronger MARL method such as centralized-critic PPO / MAPPO-style training

## Associated Commands

Main 5v5 line:

```bash
cd ~/Codes/RL/Learning-to-Play-Football-Multi-Agent-Reinforcement-Learning
source football-master/football-env/bin/activate
python -u Y_Fu/train.py --preset five_vs_five --device cpu --rollout-steps 1024 --total-timesteps 2000000 --init-checkpoint Y_Fu/checkpoints/five_vs_five/update_140.pt --pass-success-reward 0.08 --pass-failure-penalty 0.03 --pass-progress-reward-scale 0.08 --shot-attempt-reward 0.03 --attacking-possession-reward 0.0015 --final-third-entry-reward 0.04 --possession-retention-reward 0.0010 --own-half-turnover-penalty 0.015 --possession-recovery-reward 0.02 --defensive-third-recovery-reward 0.02 --opponent-attacking-possession-penalty 0.0005
```

Stage evaluation examples:

```bash
python Y_Fu/evaluate.py --checkpoint Y_Fu/checkpoints/academy_3_vs_1_with_keeper/latest.pt --episodes 10 --deterministic --compare-random --device cpu
python Y_Fu/evaluate.py --checkpoint Y_Fu/checkpoints/five_vs_five/latest.pt --episodes 10 --deterministic --compare-random --device cpu
python Y_Fu/evaluate.py --checkpoint Y_Fu/checkpoints/full_11v11_residual/latest.pt --episodes 10 --deterministic --compare-random --device cpu
```
