# Five-v-Five Failure Solution Sweep

## Purpose

This note is a focused sweep of possible solutions for the current failure mode:

- after a long `five_vs_five` PPO run
- the policy still does not look like football
- it often just pushes forward, collapses onto the ball, or drifts through full-length losing games

This is not a paper survey.

It is an LLM-assisted diagnosis and solution map for the concrete `Y_Fu` setup.

## Observed Failure Pattern

The current failure is not just "weak performance".

It has a recognizable structure:

- very low or zero `goals_for`
- `success_rate` often stays at `0.000`
- many matches go to full length `3001`
- scorelines are often `0-1`, `0-2`, `0-3`, sometimes worse
- behavior looks like motion without real football organization

The key implication is:

- the system is not merely undertrained
- it is likely optimizing the wrong easy behaviors, or failing to assign credit to the right ones

## Main Diagnosis Buckets

There are several plausible root causes.

### 1. Reward Misalignment

The dense shaping may be rewarding:

- generic progression
- safe possession
- harmless pass completion

more strongly than:

- dangerous attack creation
- shot generation
- actual scoring sequences

This can produce a policy that "moves" but does not "play".

### 2. Shared-Policy Role Collapse

The current setup uses one shared policy for the controlled outfield players.

Without an explicit identity signal, the policy can learn:

- everybody runs toward the ball
- everybody reacts similarly
- no stable off-ball role structure appears

This is a very plausible cause for the "not football" look.

### 3. Weak Credit Assignment

In football, useful decisions often pay off much later.

Examples:

- a good support run
- a spacing decision
- a decoy movement
- a safe recycle before a later attack

Plain PPO with local dense shaping can fail to connect these longer chains to later success.

### 4. Weak Warm Start

The current warm start helps the run initialize, but it may not transfer enough to `five_vs_five`.

If the initialization knows only shallow ball progression, it can drag the larger task into the same shallow behaviors.

### 5. Exploration And Optimization Mismatch

Even if the reward is not terrible, PPO can still fail if:

- exploration is too weak
- sample reuse is still not well matched to the task
- early successful patterns are too rare to reinforce

### 6. Opponent And Task Structure Limits

The built-in GRF bot is good enough as a first target, but the task still remains multi-agent and partially noisy.

Even without self-play, the current setup may be too hard unless the curriculum and rewards are better aligned.

## Solution Sweep

Below is the practical sweep of candidate solutions.

## Tier 1: Low-Cost, High-Priority Fixes

These are the first solutions worth trying because they directly target the current failure and are cheap enough to test.

### A. Reward Revision

Idea:

- reduce rewards for safe but empty play
- increase rewards for shot creation and dangerous attack completion

Why it may help:

- directly changes what PPO is optimizing
- most aligned with the observed "just runs forward" failure

What was already changed:

- reduced pass-completion incentives
- removed generic attacking-possession reward
- removed generic retention reward
- removed noisy opponent-attacking-possession penalty
- increased `shot_attempt_reward`
- increased `final_third_entry_reward`

What would count as success:

- `goals_for` no longer stays pinned to zero
- more final-third entries
- more shot attempts
- fewer full-length empty matches

Cost:

- low

Risk:

- can still produce another shaped-reward loophole

Priority:

- highest

### B. Reward Ablation Table

Idea:

- do not just hand-edit one version
- create a small matrix of reward variants and compare them

Example axes:

- with vs without retention reward
- weak vs medium shot reward
- weak vs medium turnover penalty

Why it may help:

- turns reward tuning from guesswork into controlled diagnosis

What would count as success:

- one reward family clearly produces more goals, better goal difference, or better videos

Cost:

- low to medium

Risk:

- still manual, but much more systematic than ad hoc tuning

Priority:

- highest

### C. `player_id` Input

Idea:

- keep a shared policy
- add an explicit identity input for each controlled player

Examples:

- one-hot player index
- learned player embedding

Why it may help:

- reduces homogeneous behavior
- makes role differentiation easier
- very relevant to the current "everyone behaves the same" failure mode

What would count as success:

- more stable spacing
- fewer ball-collapsing behaviors
- clearer support and coverage patterns

Cost:

- low to medium

Risk:

- if reward is still wrong, identity alone will not save the run

Priority:

- very high, immediately after reward revision if reward-only does not improve enough

### D. Better Warm Start Selection

Idea:

- choose warm starts by `five_vs_five` transfer quality, not by curriculum return

Why it may help:

- some checkpoints can look good on shaped curriculum return while being bad transfer points

What would count as success:

- early `five_vs_five` behavior improves noticeably in the first `1M ~ 2M` steps

Cost:

- low

Risk:

- helpful but usually not sufficient by itself

Priority:

- high

## Tier 2: Medium-Cost, Still Practical

These are still reasonable in this project, but they should follow the first tier.

### E. Opponent Pool / Light Self-Play

Idea:

- do not train only against the built-in bot forever
- mix built-in bot and historical checkpoints as opponents

Why it may help:

- avoids overfitting to one opponent style
- becomes useful once the policy can already play some football

Why it is not first:

- current problem is not "policy already too strong for the bot"
- current problem is "policy does not yet play meaningful football"

Cost:

- medium

Risk:

- adds complexity before the base behavior is fixed

Priority:

- medium, later than reward and `player_id`

### F. Better Curriculum Gate

Idea:

- do not spend large budgets on stages that clearly do not pass
- require measurable stage success before promotion

Why it may help:

- protects compute
- prevents false confidence from "run completed"

Why it is not the direct fix:

- it saves budget
- it does not itself teach better football

Cost:

- low

Risk:

- none, operationally very useful

Priority:

- already in use, should stay

### G. PPO Hyperparameter Sweep

Idea:

- small sweep over:
  - entropy coefficient
  - rollout length
  - minibatch count
  - update epochs
  - learning rate

Why it may help:

- some settings may preserve rare useful trajectories better

Why it is not first:

- the current failure looks more structural than purely hyperparameter-level

Cost:

- medium

Risk:

- easy to waste compute if done before fixing reward and roles

Priority:

- medium

### H. Recurrent Policy

Idea:

- give the agent memory over recent timesteps

Why it may help:

- football has partial observability
- coordination often depends on short-term temporal context

Why it is not first:

- larger code change
- harder to debug
- likely lower ROI than reward plus `player_id`

Cost:

- medium to high

Risk:

- more engineering surface area

Priority:

- medium to low

## Tier 3: Higher-Cost, More Systematic

These are interesting, but should not be the first response.

### I. Automatic Reward-Weight Search

Idea:

- keep the reward family fixed
- search reward coefficients automatically

Examples:

- random search
- grid search over a few important terms
- small Bayesian tuning loop

Why it may help:

- less subjective than pure manual tuning

Cost:

- medium

Risk:

- still depends on good evaluation criteria

Priority:

- good next systematic step after a few manual ablations

### J. Population-Based Training

Idea:

- run a small set of PPO jobs
- periodically keep the stronger ones
- mutate reward weights or PPO settings

Why it may help:

- combines tuning and training
- can discover better settings than single-run intuition

Cost:

- medium to high

Risk:

- needs more parallel compute discipline

Priority:

- later, but realistic

### K. Preference-Based Selection

Idea:

- compare short videos or checkpoints
- use human preference to decide which run is actually more football-like

Why it may help:

- useful when shaped return disagrees with visual quality

Cost:

- medium

Risk:

- introduces human review burden

Priority:

- later, but attractive for this project because "looks like football" matters

### L. Meta-Gradient Reward Optimization

Idea:

- use downstream performance to update reward coefficients

Why it may help:

- more principled than manual weight tuning

Why it is not first:

- more complex
- harder to trust
- likely too heavy for the current stage

Cost:

- high

Risk:

- high

Priority:

- low for now

## Tier 4: Bigger Algorithmic Changes

These are valid research directions, but they are not the first thing to try.

### M. Centralized Critic / MAPPO-Style Upgrade

Idea:

- keep decentralized actions
- give the critic more global multi-agent information

Why it may help:

- football is multi-agent and credit assignment is hard
- a stronger critic can stabilize learning

Why it is not first:

- bigger implementation change
- current PPO line should be pushed further before a full algorithm step-up

Cost:

- high

Risk:

- larger refactor

Priority:

- medium-long term

### N. Imitation Or Offline Warm Start

Idea:

- initialize from better demonstrations, bot trajectories, or curated play data

Why it may help:

- bypasses the worst random early exploration phase

Why it is interesting here:

- if PPO from weak warm start keeps failing, a better initialization may matter a lot

Cost:

- medium to high

Risk:

- data quality becomes the bottleneck

Priority:

- medium-long term

## What Seems Most Likely To Fix This Specific Failure

Given the current evidence, the most plausible chain is:

1. reward revision
2. reward ablation table
3. `player_id`
4. better warm-start selection
5. small PPO hyperparameter sweep
6. then light self-play or opponent pool

The reason is simple:

- the current policy does not look "almost good"
- it looks structurally wrong
- that usually points first to objective mismatch and role collapse

## What Is Probably Not The Right Immediate Conclusion

Do not assume:

- "just train longer"
- "the built-in bot is the only problem"
- "we need full self-play immediately"
- "we need a giant algorithm rewrite before trying smaller fixes"

Those may matter later, but they do not fit the current evidence as the first explanation.

## Practical Next-Step Order

For the current `Y_Fu` line, the best next sequence is:

1. finish reading the current reward-only revision run
2. compare it against the old `10M` failure pattern
3. if it still looks structurally bad, add `player_id`
4. run a small reward ablation table
5. only after that, consider opponent-pool or more systematic tuning

## Final View

Yes, there are many possible solutions.

But they are not equally likely.

The current failure most strongly suggests:

- the policy is learning the wrong dense objective
- the shared policy lacks enough identity structure

So the immediate project logic should remain:

1. fix what PPO is being pushed to optimize
2. fix how shared players are distinguished
3. only then add more sophisticated search or opponent complexity
