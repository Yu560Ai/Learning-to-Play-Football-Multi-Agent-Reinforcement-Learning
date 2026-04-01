# Academy-to-5v5 Bootstrap Plan

## Core View

The current evidence suggests that pure `5_vs_5` PPO is too hard as a from-scratch learning problem in this project.

Observed symptom:

- after long `5_vs_5` training, the policy still does not reliably show basic attacking behavior such as simple passing, support movement, and easy shot creation

Interpretation:

- this is a primitive acquisition failure, not only a late-stage coordination failure
- the agent is struggling to learn easy football actions inside a long-horizon multi-agent environment
- therefore, a smaller teaching scenario is justified

This makes the Academy stages useful again, but only in a specific role:

- Academy is for teaching attacking primitives
- `5_vs_5` is for learning transfer, transition behavior, spacing, and robust decision making

So the intended strategy is:

1. learn basic attack structure in Academy
2. transfer the best Academy checkpoint into `5_vs_5`
3. spend the larger training budget in `5_vs_5`
4. judge success in `5_vs_5`, not in Academy

## Why This Approach Makes Sense

### Why `5_vs_5` from scratch is hard

In full `5_vs_5`, PPO must solve several problems at once:

- ball control
- movement and orientation
- discovering passing
- discovering shooting
- learning support runs
- learning spacing
- learning transition defense
- coordinating multiple controlled players

That is a high exploration burden with weak and delayed credit assignment.

If the policy cannot even learn easy passes in `5_vs_5`, then expecting the same setup to discover clean multi-agent attacking structure by itself is too optimistic.

### Why Academy can help

Academy tasks make the learning problem much easier:

- shorter episodes
- denser success signals
- clearer attacking geometry
- less irrelevant state variation
- more repeated exposure to pass-and-finish situations

This is exactly the kind of setting that can teach:

- when to release the ball
- how to support the ball carrier
- how to face the goal before shooting
- how to convert numerical advantage into a shot

### Why Academy alone is not enough

Academy scenarios are heavily scripted and much narrower than `5_vs_5`.

Risks:

- overfitting to one attack pattern
- learning to pass only in a fixed local geometry
- weak transition behavior after losing the ball
- poor defensive recovery
- poor use of open space outside the scenario template

So Academy should be treated as a bootstrapping stage, not the main destination.

## Training Principle

Use Academy to solve primitive behavior, then use `5_vs_5` to generalize and stabilize it.

Practical implication:

- do not abandon Academy if `5_vs_5` still looks primitive
- do not spend most total compute in Academy after the primitive is clearly present

The correct balance is:

- enough Academy to create a real passing-and-shooting prior
- most total steps in `5_vs_5`

## Proposed Curriculum

### Stage 1: `academy_run_to_score_with_keeper`

Purpose:

- teach direct ball progression
- teach goal-oriented movement
- teach finishing against the keeper

Desired behavior:

- agent moves decisively toward goal
- agent can beat the keeper in simple 1-player situations

### Stage 2: `academy_pass_and_shoot_with_keeper`

Purpose:

- teach the most basic pass-then-finish pattern
- teach that passing can be better than dribbling

Desired behavior:

- ball carrier passes instead of forcing bad dribbles
- receiver positions to finish
- policy can convert obvious 2-player attacks reliably

### Stage 3: `academy_3_vs_1_with_keeper`

Purpose:

- teach simple numerical-advantage attack structure
- teach one more layer of support and decision making

Desired behavior:

- policy uses the extra attacker instead of tunnel-vision dribbling
- attack can adapt when the first lane is blocked
- ball circulation leads to a shot with reasonable consistency

### Stage 4: `5_vs_5`

Purpose:

- transfer Academy primitives into a more realistic game
- learn spacing, recovery, transitions, and more robust decision making

Desired behavior:

- obvious pass opportunities are used
- teammates spread instead of clustering
- attack continues after the first action instead of collapsing
- policy begins to create shots from live play

## Gating Rule

Do not leave an Academy stage only because training finished.

Leave a stage only if the behavior is visibly present and the evaluation threshold is acceptable.

Suggested practical gate:

- keep stage-specific evaluation thresholds
- also inspect videos for qualitative behavior
- only transfer checkpoints that clearly show the target primitive

Important:

- "run completed" is not the same as "stage passed"
- the handoff checkpoint should be chosen for behavior quality, not just latest timestamp

## What To Transfer

When moving into `5_vs_5`, transfer the best Academy checkpoint, not automatically `latest.pt`.

Checkpoint selection should favor:

- visible passing behavior
- stable shot creation
- less random dribbling
- repeatable attack structure across several episodes

If an earlier checkpoint shows cleaner football than a later one, use the earlier checkpoint.

## Budget Recommendation

The total budget should still favor `5_vs_5`.

Recommended allocation shape:

- short-to-moderate Academy warm-up
- large `5_vs_5` block after transfer

In other words:

- Academy teaches the skill
- `5_vs_5` teaches where and when to use the skill

## What To Measure

Win rate alone is not enough for early curriculum decisions.

Track these signals, especially in `5_vs_5`:

- completed pass count
- pass completion rate
- shots per episode
- goals per episode
- final-third entries
- possession retention after receiving the ball
- qualitative video evidence of support runs

These metrics matter because the near-term question is not only "does the team win?" but also:

- did the policy actually learn to pass?
- did Academy transfer into live `5_vs_5` behavior?

## Main Risks

### Risk 1: Overtraining in Academy

If Academy training is too long, the policy may become specialized to scripted situations and transfer poorly.

Response:

- use Academy as a warm-up, not the main compute sink
- move to `5_vs_5` once the primitive is real

### Risk 2: False positive from one good video

A single good episode can be misleading.

Response:

- evaluate over multiple seeds
- compare several checkpoints
- use both metrics and video

### Risk 3: Transfer failure even after Academy success

The policy may learn passing in Academy but fail to use it in `5_vs_5`.

Response:

- keep measuring pass-related metrics in `5_vs_5`
- revise `5_vs_5` shaping if the transferred policy immediately forgets passing
- compare transfer against `5_vs_5` from scratch, not only against itself

## Concrete Experiment Plan

### Experiment A: Establish the primitive

Goal:

- produce one checkpoint from Academy that clearly demonstrates passing and shot creation

Steps:

1. train `academy_pass_and_shoot_with_keeper`
2. evaluate several checkpoints, not only `latest.pt`
3. continue with `academy_3_vs_1_with_keeper`
4. again select the best checkpoint by behavior and evaluation, not by recency

### Experiment B: Test transfer into `5_vs_5`

Goal:

- determine whether Academy improves early `5_vs_5` behavior relative to scratch PPO

Steps:

1. initialize one `5_vs_5` run from the best Academy checkpoint
2. initialize another `5_vs_5` run from scratch
3. compare them at the same step budgets
4. inspect pass count, pass completion, shot creation, and video behavior

Success condition:

- transferred run shows clearer passing and better attack construction early in training

### Experiment C: Decide whether Academy is worth keeping

Keep Academy in the pipeline if it does at least one of these:

- improves early `5_vs_5` passing behavior
- improves shot creation in `5_vs_5`
- improves sample efficiency relative to scratch PPO

Reduce or remove Academy if:

- it does not improve `5_vs_5` behavior
- it only produces scripted-looking patterns with no transfer
- it consumes too much budget compared with direct `5_vs_5` training

## Bottom Line

The current working hypothesis should be:

- `5_vs_5` from scratch is too difficult for learning basic football primitives in the current PPO setup
- Academy is likely useful for bootstrapping those primitives
- Academy is not the final objective and should not dominate total training time
- the real test is whether Academy creates better `5_vs_5` passing and attack behavior

This is the practical stance:

- keep Academy
- use it deliberately
- transfer early enough
- evaluate success in `5_vs_5`
