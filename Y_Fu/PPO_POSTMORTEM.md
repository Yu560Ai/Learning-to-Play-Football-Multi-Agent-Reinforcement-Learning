# PPO Postmortem

## Purpose

This note summarizes what the current `Y_Fu` PPO line actually taught us.

It is meant to answer three questions:

1. what did the PPO runs learn well enough to keep?
2. what repeatedly failed, even after more training or small fixes?
3. how should the next reward design move closer to real football behavior and farther from loophole exploitation?

This is not a generic PPO note.

It is a postmortem for the current `academy -> 5_vs_5` line in this repository.

## Bottom Line

The strongest current conclusion is:

- PPO did learn some local structure
- but it did not learn stable match-winning football
- the main blocker is not runtime instability
- the main blocker is that the current objective still rewards too many cheap local behaviors and not enough true attack quality

The current evidence does **not** support:

- "just add more timesteps"
- "player identity alone fixes it"
- "academy warm-up is enough by itself"

The current evidence **does** support:

- reward redesign is still central
- stage gates must use football outcomes and video, not shaped return
- the reward should be tied more closely to real football process metrics
- the reward should become harder to game with harmless possession or empty movement

## What The Runs Showed

### Academy Runs

`academy_pass_and_shoot_with_keeper` did produce some useful primitives:

- better short-horizon progression
- some pass-to-shot intent
- some shaped-return improvement

But the evidence also showed:

- `avg_return` improved much more than `avg_score_reward`
- deterministic evaluation still often failed to score
- later checkpoints could become worse than earlier ones

That means Academy was useful as a weak bootstrap, not as a proven solved handoff stage.

### `5_vs_5` Scratch Or Near-Scratch PPO

The `five_vs_five` line consumed a meaningful budget and still produced:

- near-zero goals for
- repeated full-length `3001`-step matches
- losses such as `0-1`, `0-2`, `0-3`, worse in many cases
- movement without real football structure

This is the core negative result.

It tells us the current setup is not merely undertrained.

It is optimizing the wrong easy behaviors or assigning credit too weakly to the right ones.

### Reward Revision

The reward-only revision was directionally correct:

- less generic possession reward
- less retention reward
- stronger shot and final-third emphasis

But it still did not create enough actual attack quality.

This means the reward revision was necessary, but not sufficient.

### `player_id`

The `player_id` run helped behavior diagnostics:

- less extreme rightward collapse
- somewhat better pass and shot rates for a while

But final evaluation still showed:

- `win_rate = 0.000`
- `avg_goal_diff = -1.400`
- matches still reaching full length

So `player_id` reduced one structural pathology, but did not fix the objective.

## Stable Failure Pattern

Across the current PPO line, the repeated failure pattern is:

1. shaped return can improve
2. score reward and winning do not improve enough
3. policy still produces full-length low-event matches
4. behavior diagnostics still show degenerate action mixtures

This is exactly the pattern you get when the reward contains events that are:

- frequent
- easy to trigger
- only weakly tied to winning

and when those events outweigh rarer but truly useful football behavior.

## Root Cause Ranking

Current ranking of likely causes:

1. reward misalignment
2. weak long-horizon credit assignment
3. weak successful-trajectory coverage
4. shared-policy role collapse
5. curriculum handoff not strong enough

### 1. Reward Misalignment

This is still the top cause.

The current shaping logic in [envs.py](/home/yuhan/Codes/RL/Learning-to-Play-Football-Multi-Agent-Reinforcement-Learning/Y_Fu/yfu_football/envs.py) is built from local events:

- pass success
- pass progress
- shot attempt
- final-third entry
- possession retention
- possession recovery
- turnover penalty

This family is reasonable for a first draft.

But it is still vulnerable to reward gaming because several terms are:

- local
- frequent
- only loosely connected to true chance creation

### 2. Weak Long-Horizon Credit Assignment

In real football, useful decisions often matter several actions later:

- an off-ball support run
- choosing not to force a pass
- delaying for a better passing lane
- maintaining rest defense while attacking

Plain PPO with only local event shaping does not connect these long chains well.

### 3. Weak Successful-Trajectory Coverage

The issue is not just "entropy is too low".

The deeper issue is:

- the built-in opponent is fixed
- successful attacks are still rare
- the policy keeps replaying mediocre but repeatable local patterns

So the exploration problem is really a **useful trajectory coverage** problem.

### 4. Shared-Policy Role Collapse

This is real, but secondary.

The `player_id` experiment confirms that role information matters.

It does not confirm that role collapse was the main blocker.

### 5. Curriculum Handoff Strength

Academy helped, but the best handoff checkpoint still did not clearly pass its own stage gate.

So `5_vs_5` was starting from weak primitives, not stable football skills.

## What Reward Terms Are Still Dangerous

These are the most suspicious reward families when the goal is real `5_vs_5` football.

### Dangerous Family A: Generic Possession

Examples:

- attacking possession reward
- possession retention reward

Why dangerous:

- extremely frequent
- low causal connection to chance creation
- easy to exploit with harmless circulation or standing in a favorable zone

Current rule:

- keep them at `0.0`

### Dangerous Family B: Pass Success As An End In Itself

A pass is not inherently good.

A pass is good if it improves the attack.

Why dangerous:

- pass completion is easy to reward
- harmless passing can dominate learning
- it can substitute for actual chance creation

Current rule:

- pass reward should stay weaker than shot-quality or dangerous-entry signals

### Dangerous Family C: Team-Level Blame Without Attribution

Examples:

- opponent attacking possession penalties
- large global penalties for broad defensive states

Why dangerous:

- these events are noisy in multi-agent play
- the trained players may not be clearly responsible
- the reward can punish the policy for states it did not meaningfully control

Current rule:

- avoid or heavily downweight such penalties

## Why Real-Football Metrics Matter

The current reward family is mostly built from easy environment events.

That is practical, but it is not how real teams judge whether an attack or defensive phase was good.

Real football coaching evaluation is usually closer to:

- did possession become a real chance?
- did we enter dangerous space with support?
- did we protect ourselves against transition?
- did we lose the ball in a dangerous way?
- did the team stay structurally useful off the ball?

Those ideas translate better into RL if the reward is anchored to **process quality**, not just to raw possession or local movement.

## A More Football-Like Reward Philosophy

The reward should move from:

- "did something locally positive happen?"

toward:

- "did the team improve the quality of the attack or the defensive situation in a way that a coach would actually value?"

This suggests five reward families.

### 1. Chance Creation

This is the most important attacking family.

Coach-like metrics:

- shots created
- shots created after a pass sequence
- touches or actions in highly dangerous areas
- repeated creation of finishable situations

RL interpretation:

- reward attack completion, not just attack continuation
- reward should rise more for entering a true shooting context than for generic forward movement

### 2. Progression Quality

Progression is useful, but only if it improves the attack.

Coach-like metrics:

- final-third entries
- line-breaking progression
- progression that keeps support options alive

RL interpretation:

- progression reward should be conditional and weaker than chance-creation reward
- not every forward move deserves reward

### 3. Possession Value, Not Possession Volume

Real teams do not value possession equally in all contexts.

Coach-like metrics:

- possession that leads to chance creation
- possession that survives pressure and improves field position
- possessions that avoid dangerous turnovers

RL interpretation:

- do not reward raw possession duration
- instead reward possession sequences that produce improved attacking state

### 4. Transition Protection

This is the biggest thing Academy cannot teach well.

Coach-like metrics:

- own-half turnovers
- opponent transition starts after our attack
- speed of defensive recovery
- how often opponent reaches our dangerous zones after our loss

RL interpretation:

- small penalties for dangerous turnovers make sense
- avoid broad global defensive blame
- penalize the most clearly attributable transition mistakes first

### 5. Team Structure

Real teams care about spacing and support shape even when nobody touches the ball.

Coach-like metrics:

- support options around the ball
- not collapsing all players onto the ball
- useful width and depth

RL interpretation:

- this is hard to reward directly with the current code
- but it should at least be monitored, and later converted into reward only if the signal is robust

## A Better Reward Design Rule

For the next PPO line, each shaping term should pass this filter:

1. Is it hard to trigger with empty play?
2. Is it closer to winning than generic possession is?
3. Is responsibility reasonably attributable to the controlled players?
4. Is it infrequent enough that it will not dominate the sparse task anchors?
5. Would a human coach consider it a genuinely good football event?

If the answer is "no" to most of these, that term should not be in the reward.

## Concrete Football-Like Metrics To Track

These should be logged even before they become reward terms.

### Attacking Metrics

- shots per episode
- final-third entries per episode
- attacks that end in a shot
- goals for
- mean goal difference
- fraction of possessions that cross the attacking threshold and then produce a shot

### Ball Security Metrics

- own-half turnovers per episode
- turnovers within a fixed horizon after entering the final third
- fraction of passes that improve ball position meaningfully

### Transition Metrics

- opponent final-third entries after our possession loss
- opponent shots after our own-half turnover
- recoveries in defensive third

### Structure Metrics

- mean distance of non-ball players to the active player
- width proxy: spread of controlled outfield x/y positions
- collapse proxy: fraction of time too many controlled players cluster near the ball

### Event Quality Metrics

- shot rate is not enough
- distinguish "shot happened" from "shot came from a useful attacking sequence"

This is where the current tracking is still too shallow.

## Reward Terms That Are Safer In The Next Iteration

If we stay within the current code style, the safer shaping family is:

- sparse anchors: `scoring,checkpoints`
- modest final-third entry reward
- modest shot-attempt reward
- modest own-half turnover penalty
- modest possession recovery reward

and keep at or near zero:

- attacking possession reward
- possession retention reward
- opponent attacking possession penalty
- oversized pass success reward

## Reward Terms Worth Adding Later, But Only Carefully

These are more football-like, but they require better implementation than the current code exposes.

### Conditional Pass Reward

Reward the pass only if it leads to:

- final-third entry
- a shot within a short horizon
- a materially better attacking state

This is much better than rewarding pass completion itself.

### Sequence Completion Reward

Reward short attack chains, for example:

- recover ball -> progress -> final-third entry
- pass -> support -> shot

This is closer to how coaches value possessions.

### Transition-Failure Penalty

Penalize:

- our own-half turnover followed by fast opponent danger

not merely:

- opponent has the ball somewhere dangerous

This makes the penalty more causal and less noisy.

### Spacing Or Support Reward

Only add this if the metric is reliable.

Naive spacing rewards are easy to break and can become another loophole.

## What Should Change In Practice

The next PPO improvement loop should not be:

- change one coefficient
- run all night
- hope

It should be:

1. define 3 to 5 reward variants
2. log football-like metrics, not only return
3. evaluate early at `250k ~ 500k env steps`
4. kill runs that still show:
   - zero goals
   - full-length empty games
   - degenerate action mixtures
5. only extend the reward family that improves both:
   - football-like metrics
   - actual match outcomes

## Recommended Next Reward Ablation Table

Use a small matrix, not one hand-edited run.

### Variant A: Minimal Attack Reward

- `pass_success_reward = 0.02`
- `pass_failure_penalty = 0.02`
- `pass_progress_reward_scale = 0.02`
- `shot_attempt_reward = 0.08`
- `final_third_entry_reward = 0.08`
- `possession_recovery_reward = 0.01`
- `defensive_third_recovery_reward = 0.015`
- `own_half_turnover_penalty = 0.02`
- others at `0.0`

Goal:

- strongly bias toward attack completion

### Variant B: Conservative Transition Version

- same as Variant A
- slightly stronger own-half turnover penalty
- slightly weaker pass reward

Goal:

- see if the current line needs more transition discipline

### Variant C: Conditional-Progress Proxy

- keep final-third and shot reward
- keep pass reward minimal
- slightly stronger pass-progress scale than Variant A

Goal:

- test whether line-breaking style progression matters more than plain pass success

## What Should Not Be Overinterpreted

Two things should be treated carefully.

### 1. More Randomness

Adding randomness blindly is not the answer.

More randomness helps only if it increases useful trajectory coverage.

If it mainly produces junk actions, it will just slow PPO further.

### 2. More Dense Terms

Adding more reward terms is not automatically more realistic.

Often it does the opposite.

The right direction is:

- fewer
- more causal
- more coach-like
- harder to exploit

## Final Conclusion

The current PPO line failed in a useful way.

It showed that:

- local shaped progress is learnable
- real `5_vs_5` football is not emerging from the current objective
- reward redesign still matters more than extra budget

The next reward should behave less like:

- "credit every locally nice-looking action"

and more like:

- "credit possessions that become real danger and protect the team against the next transition"

That is the cleanest way to make the reward more realistic and less vulnerable to loopholes.
