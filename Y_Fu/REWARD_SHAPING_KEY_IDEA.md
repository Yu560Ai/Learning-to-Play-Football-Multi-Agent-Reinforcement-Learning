# Reward Shaping Key Idea

## Core Point

In PPO, reward shaping does not only help the value network.

It changes the full optimization loop:

```text
reward
-> return / GAE
-> advantage
-> policy update
-> value update
-> new behavior
-> new data
-> new reward
```

So changing reward shaping means changing what the whole PPO system is trying to learn.

## The Actual PPO Data Flow

At time step `t`, the environment produces a training reward:

```text
r'_t = r_env_t + r_shape_t
```

where:

- `r_env_t` is the original environment reward
- `r_shape_t` is the extra shaping reward

Then PPO uses that reward in the full loop:

1. Collect rollout data

```text
(obs_t, action_t, logprob_t, value_t, reward_t, done_t)
```

2. Compute TD residual

```text
δ_t = r'_t + γ * V(s_{t+1}) * (1 - done_t) - V(s_t)
```

3. Compute GAE

```text
A_t = δ_t + γλ(1-done_t)δ_{t+1} + ...
```

4. Compute return target

```text
R_t = A_t + V(s_t)
```

5. Update actor using `A_t`

6. Update critic using `R_t`

So:

- actor is changed through `advantage`
- critic is changed through `return`
- both are changed by reward shaping

## Revision To The Earlier Idea

The earlier simple rule was:

- keep scoring and conceding as the core reward

That is directionally correct, but incomplete.

For this football project, a better version is:

- keep task-success signals as anchors
- do not rely on goals alone in early learning
- use dense rewards that are closer to scoring and less noisy than raw possession outcomes

## Why Goal Reward Alone Is Not Enough

You are right:

- goals are very sparse
- an untrained model may almost never score
- if reward is too sparse, PPO may not see enough successful examples to build a useful advantage signal

So "goal reward is the true objective" does not mean:

- use only goal reward

It means:

- goal-related task success should remain the reference objective
- shaping should help the policy approach that objective instead of replacing it

## Why Raw Ball-Loss Or Conceded-Goal Penalties Can Be Bad

You are also right that:

- losing the ball is noisy
- conceding is even noisier at team level
- in multi-agent football, not every turnover or conceded goal is attributable to the currently learned player behavior

That means some dense defensive penalties can inject noisy gradients.

If they are too strong, PPO may learn:

- risk-avoidant but passive play
- safe possession without real attack
- behavior that avoids blame rather than solves the task

## Better Reward Design Principle

Use three layers of reward.

### Layer 1: Sparse task anchors

These define the real task direction:

- scoring a goal
- conceding a goal

These should stay in the system.

But they should not be the only learnable signal early on.

### Layer 2: Near-causal dense events

These are the most useful shaping terms.

Good examples:

- entering dangerous attacking zones
- successful forward pass into a better position
- creating a shot attempt
- recovering possession in a meaningful defensive area
- progressing the ball toward a high-value attacking state

These are much closer to actual scoring than generic possession rewards.

### Layer 3: Weak or noisy support signals

These should be small, or sometimes removed.

Examples:

- generic possession retention reward
- generic turnover penalty
- large concede-based blame terms

These are often too noisy or too weakly causal.

## Practical Rule

A shaping term is useful if:

- it happens more often than goals
- it is causally closer to scoring or preventing danger
- it is attributable enough to the learned behavior

A shaping term is dangerous if:

- it is frequent but weakly tied to winning
- it mostly encourages safe but empty behavior
- it punishes the agent for team-level outcomes it cannot yet control well

## What This Means For Our Current Training

The current academy results suggest:

- the policy can increase shaped return
- but it still does not solve the task

That usually means one of these:

1. the shaping signal is too easy to exploit without finishing the task
2. the shaping terms are not close enough to real scoring behavior
3. the penalties are noisy enough to wash out useful credit assignment

So the next reward revision should not be:

- "remove all shaping and only reward goals"

It should be:

- keep goal/concede as anchors
- strengthen dense rewards that are closer to shot creation and real attack completion
- weaken noisy, weakly attributable penalties

## Safer Standard Reward Strategy

For this project, a better standard shaping template is:

1. Keep:

- goal reward
- concede penalty
- GRF checkpoint-style progression reward

2. Strengthen only if they are causally useful:

- shot attempt reward
- final-third entry reward
- pass-progress reward
- possession recovery in meaningful zones

3. Reduce if they appear noisy:

- generic possession retention reward
- generic turnover penalty
- heavy concede-side shaping beyond the basic goal-against penalty

## Key Design Constraint

Dense shaping should help the agent approach the real task.

Dense shaping should not become an easier substitute objective.

If this happens, PPO will optimize the substitute objective well and still fail the actual football task.

## Standard Reward Engineering + Ablation

For this project, the standard approach is not meta-learning first.

The standard approach is:

1. reward engineering
2. ablation
3. evaluation on the real task

### Reward engineering

Reward engineering means:

- decide which events should be rewarded
- decide which events should be penalized
- decide how strong each term should be

This should be guided by task structure, not by arbitrary numbers.

In football, that means asking:

- is this signal closer to actually scoring?
- is this signal less noisy than raw concede events?
- can this signal be attributed to the learned behavior?

### Ablation

Ablation means:

- do not change many reward terms at once without control
- remove or weaken one small group of terms
- compare the result against the current baseline

This is important because reward terms interact.

For example:

- if pass reward is too strong, the model may over-value safe circulation
- if turnover penalty is too strong, the model may become overly passive
- if shot reward is too weak, the model may never learn to finish attacks

So when a run fails, the right question is not:

- "should we try random new reward numbers?"

The right question is:

- "which reward term is causing the wrong behavior, and how do we isolate it?"

### Example ablation workflow

1. keep the current reward setup as the baseline
2. form one clear hypothesis
3. change only the relevant reward terms
4. rerun evaluation
5. compare real task metrics and video behavior

Example:

- hypothesis: `possession_retention_reward` encourages safe but empty control
- ablation: set `possession_retention_reward = 0.0`
- compare:
  - `win_rate`
  - `avg_goal_diff`
  - representative video behavior

### Why this is the standard method here

This approach is:

- simpler
- more interpretable
- cheaper
- easier to debug

than introducing meta-gradient reward tuning or reward-learning systems at this stage.

### What meta-learning would mean instead

That would be something like:

- automatically learning reward weights
- using an outer optimization loop for reward coefficients
- meta-gradient RL
- population-based tuning of reward weights

Those are real methods, but they are not the first practical move for this repo.

### Practical rule for this project

If `five_vs_five` fails, the next reward revision should be done by:

1. identifying the most suspicious shaping term
2. ablating or weakening it
3. changing as few reward terms as possible in one round
4. validating with real football metrics rather than shaped return alone

## Working Rule For Future Edits

When we revise reward shaping, the order should be:

1. Check whether `five_vs_five` improves under the current setup
2. If not, revise reward so that high-frequency rewards are more tightly connected to real scoring chances
3. Reduce noisy team-level blame terms before adding more penalties
4. Only then consider structural changes like `player_id`

## References In `refs/`

These ideas are aligned with the local papers already in the repo:

- [Google Research Football.pdf](/home/yuhan/Codes/RL/Learning-to-Play-Football-Multi-Agent-Reinforcement-Learning/refs/Google%20Research%20Football.pdf)
  - supports checkpoint-style shaping because pure scoring reward is too sparse on hard football tasks

- [The Surprising Effectiveness of PPO in Cooperative.pdf](/home/yuhan/Codes/RL/Learning-to-Play-Football-Multi-Agent-Reinforcement-Learning/refs/The%20Surprising%20Effectiveness%20of%20PPO%20in%20Cooperative.pdf)
  - supports standard PPO/GAE training flow and careful PPO update settings in cooperative tasks

- [Celebrating Diversity in Shared Multi-Agent.pdf](/home/yuhan/Codes/RL/Learning-to-Play-Football-Multi-Agent-Reinforcement-Learning/refs/Celebrating%20Diversity%20in%20Shared%20Multi-Agent.pdf)
  - supports the concern that shared policies may learn homogeneous behavior in complex coordination tasks
