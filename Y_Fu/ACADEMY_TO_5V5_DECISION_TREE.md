# Academy To 5v5 Decision Tree

## Purpose

This file answers two practical questions for the current `Y_Fu/` line:

1. after the current `five_vs_five` PPO run finishes, when should we go directly into offline RL pilot, and when should we go back to Academy?
2. what can Academy realistically teach, and what behaviors only become learnable in real `5_vs_5` play?

This document is operational.

It is not trying to redefine the whole roadmap.

It is meant to help choose the next run after the current PPO job ends.

## Current Repo-Level Conclusion

From the current repo documents:

- Academy is still useful as a PPO bootstrap stage
- the current best warm-up handoff checkpoint is still `academy_pass_and_shoot_with_keeper/update_10.pt`
- `five_vs_five` is the main current stage
- offline RL is a later `5_vs_5` refinement stage, not a replacement for Academy warm-up

This is consistent with:

- [TRAINING_STAGE_LOG.md](/home/yuhan/Codes/RL/Learning-to-Play-Football-Multi-Agent-Reinforcement-Learning/Y_Fu/TRAINING_STAGE_LOG.md)
- [THREE_DAY_5V5_TRAINING_PLAN.md](/home/yuhan/Codes/RL/Learning-to-Play-Football-Multi-Agent-Reinforcement-Learning/Y_Fu/THREE_DAY_5V5_TRAINING_PLAN.md)
- [MARL_ROADMAP_SPEC.md](/home/yuhan/Codes/RL/Learning-to-Play-Football-Multi-Agent-Reinforcement-Learning/Y_Fu/MARL_ROADMAP_SPEC.md)

## Main Recommendation

Do not interrupt the current `five_vs_five` PPO run.

After it finishes:

- always evaluate the resulting checkpoints
- always allow a small offline RL pilot from the best `5_vs_5` checkpoint
- but only commit to a larger next online PPO run in `five_vs_five` if the current run shows meaningful transfer quality

If the run still looks weak, the next online PPO cycle should go back to Academy in this order:

1. short `academy_run_to_score_with_keeper` only if direct carry-and-finish looks absent
2. `academy_pass_and_shoot_with_keeper` as the main Academy stage
3. `academy_3_vs_1_with_keeper` as the last transfer filter
4. transfer to `five_vs_five`

At this stage, `academy_run_to_score_with_keeper` is not the highest-priority Academy stage for this branch.

It is useful as a solo attacking drill, but it is not the main multi-agent stepping stone.

Practical budget rule:

- keep Stage 1 short
- let Stage 2 absorb most Academy compute
- do not let Stage 3 become an endless sink if transfer quality is not improving

## Decision Tree After The Current PPO Run

### Case A: Strongest Acceptable Outcome

If the finished `five_vs_five` run shows most of these:

- `win_rate >= 0.20 ~ 0.25`
- `avg_goal_diff` is still negative, but clearly improved versus older checkpoints
- videos show repeated purposeful passing or support
- some attacks are reproducible and not purely chaotic

Then:

- use the best `five_vs_five` checkpoint for offline RL pilot
- do not immediately go back to Academy
- let offline RL test whether the current `5v5` policy already contains enough useful structure to refine

Recommended next step:

1. evaluate best and weaker `5v5` checkpoints
2. launch offline RL pilot
3. decide later whether another Academy cycle is still needed

### Case B: Borderline Outcome

If the finished `five_vs_five` run shows:

- still losing badly
- but some visible cooperative signals exist
- the behavior is not fully random
- there are occasional useful passes or final-third entries

Then:

- still run the offline RL pilot
- but do not commit to full `50M` offline data collection yet
- after the pilot, prepare the next online PPO run to go back through Academy

Recommended next step:

1. run offline RL pilot from best `5v5` checkpoint
2. in parallel, plan the next online PPO restart from Academy
3. if the pilot does not help, return to Academy before the next long PPO budget

### Case C: Clearly Bad Outcome

If the finished `five_vs_five` run still shows most of these:

- `win_rate` near `0`
- heavily negative `avg_goal_diff`
- videos still look mostly random, passive, or collapsed around the ball
- almost no reproducible team attack patterns

Then:

- still allow a small offline RL pilot if the pipeline itself needs to be validated
- but do not treat that checkpoint as good enough for large-scale offline collection
- the next meaningful online PPO run should return to Academy first

Recommended next step:

1. run only a small offline RL pilot for pipeline validation
2. restart online PPO from Academy
3. transfer back into `five_vs_five` only after Academy looks better

## Which Academy Stages Matter Most

### Highest Priority Academy Stages For This Branch

1. `academy_pass_and_shoot_with_keeper`
2. `academy_3_vs_1_with_keeper`
3. `academy_run_to_score_with_keeper` only as a supporting short stage when needed

Why these two matter:

- they teach multi-player attacking cooperation
- they are the closest Academy bridge to the shared-policy multi-agent setup
- they are the current intended stepping stones into `five_vs_five`

### Lower Priority Right Now

`academy_run_to_score_with_keeper`

Why it is lower priority:

- it is primarily a solo attack drill
- it teaches dribble-to-shot and pressure handling
- it does not directly teach multi-player coordination

It can still be useful as a support stage, but it should not replace the two cooperative Academy stages above.

## What Academy Can Realistically Teach

Academy can teach local or short-horizon football primitives very well.

### `academy_run_to_score_with_keeper`

Best for:

- ball carrying under pressure
- finishing against a keeper
- deciding when to shoot rather than over-carry
- staying composed while being chased from behind

Weak at teaching:

- team support
- passing choices
- defensive recovery shape
- multi-player spacing

### `academy_pass_and_shoot_with_keeper`

Best for:

- pass-to-shot sequences
- using a free teammate instead of forced solo dribble
- recognizing that passing can create a cleaner shot
- short cooperative attack patterns

Weak at teaching:

- repeated possession circulation
- choosing among multiple support options
- recovery after turnover
- larger team spacing

### `academy_3_vs_1_with_keeper`

Best for:

- selecting between multiple teammates
- support positioning in a small attacking group
- avoiding the single obvious passing lane
- learning 3-player attacking cooperation instead of 2-player only

Weak at teaching:

- full transition defense
- retreat shape after losing the ball
- defending wide spaces
- balancing attack and defense over a long episode

## What Only `5_vs_5` Can Really Teach

There are behaviors that Academy can prepare, but not fully teach.

These require actual `five_vs_five` play.

### 1. Transition Defense

This includes:

- recovering after a turnover
- retreating into useful defensive positions
- not letting all players stay high after losing the ball

Academy small attack drills do not expose this at the right scale.

### 2. Team Shape

This includes:

- width
- depth
- not collapsing multiple players onto the ball
- keeping at least some support structure behind the current carrier

Academy can hint at support behavior, but real shape only becomes meaningful in `5_vs_5`.

### 3. Possession Under Full-Stage Pressure

This includes:

- holding the ball through multiple action phases
- resetting an attack when the first option is gone
- not confusing "one successful pass" with "good possession"

Academy drills are too short and too local to fully teach this.

### 4. Defensive-Attacking Tradeoff

This includes:

- not overcommitting numbers forward
- deciding when to keep support behind the ball
- avoiding own-half turnovers under realistic pressure

This is a true small-team football behavior and has to be learned in `five_vs_five`.

### 5. Longer-Horizon Coordination

This includes:

- several pass decisions in one possession
- circulation before penetration
- recovery and re-attack patterns

Academy usually ends too early to force this.

## Practical Interpretation

The right way to think about Academy is:

- Academy teaches attack primitives
- `five_vs_five` teaches football

More precisely:

- Academy can teach "how to use a teammate"
- `five_vs_five` teaches "how a team should behave over a whole possession and after losing the ball"

So yes, some behaviors can only really be learned in `5v5`.

That is expected.

It does not mean Academy is unnecessary.

It means Academy should teach the reusable short-horizon pieces before `5v5` teaches the longer-horizon structure.

## Recommended Next Online PPO Line

If the current `five_vs_five` run finishes weakly, the next online PPO line should be:

1. `academy_pass_and_shoot_with_keeper`
   - continue or restart from the strongest checkpoint
2. `academy_3_vs_1_with_keeper`
   - transfer from the stronger `2-agent` checkpoint
3. `five_vs_five`
   - transfer from the stronger `3-agent` checkpoint

This is the most consistent Academy-to-`5v5` route for the current branch.

## Recommended Offline RL Positioning

Offline RL should be used like this:

- not as a substitute for Academy
- not as a way to skip `five_vs_five` PPO entirely
- but as a refinement stage once a `5v5` policy has at least some useful structure

So the operational order is:

1. Academy PPO if needed
2. `5_vs_5` PPO
3. `5_vs_5` offline RL pilot
4. larger offline RL only if the pilot looks justified

## Final Decision Rule

If you want one short rule:

- go back to Academy to teach cooperative attacking primitives
- expect real team defense, shape, and transition behavior to be learned only in `five_vs_five`

So:

- `academy_pass_and_shoot_with_keeper` and `academy_3_vs_1_with_keeper` are the right Academy stages
- some important football behaviors absolutely must be learned in `5v5`
- therefore the correct strategy is not Academy-only or `5v5`-only
- it is Academy preparation followed by `5v5` football learning
