# PPO and Offline RL Integration Plan

## Main Conclusion

The most reasonable combination for the current `Y_Fu` stack is:

1. use Academy to teach primitives through PPO
2. transfer that PPO policy into `5_vs_5`
3. keep online PPO as the main method for live adaptation in `5_vs_5`
4. use offline RL to improve sample efficiency and stabilize policy improvement on `5_vs_5`
5. optionally transfer the offline policy back into PPO for another online phase

In short:

- Academy is the primitive teacher
- PPO is the online adaptation method
- offline RL is the replay-and-refinement method

## Why This Is The Right Split

### Why not only PPO

Current evidence suggests that pure `5_vs_5` PPO from scratch is struggling to learn basic football behavior such as simple passing and attack construction.

That means the online problem is still too hard at the primitive-learning stage.

### Why not only offline RL

Offline RL can only improve from the behavior and rewards contained in the dataset.

If the dataset is weak, narrow, or missing the target behavior, offline RL cannot invent the full game from nothing.

So offline RL should not replace the online curriculum. It should exploit it.

### Why Academy still matters

Academy gives:

- shorter episodes
- denser credit assignment
- repeated exposure to pass-and-shoot situations
- better chance to teach passing before `5_vs_5`

### Why `5_vs_5` still matters most

Only `5_vs_5` teaches:

- transition behavior
- spacing under pressure
- recovery after losing the ball
- repeated multi-step attacks in a realistic live game

So the real target remains `5_vs_5`.

## Important Code-Level Facts

### 1. PPO and IQL already share compatible observation and network style

Both the current PPO and IQL lines use:

- shared per-player observations
- `extracted` representation for the main attacking stages
- CNN encoder plus MLP body
- `19`-action discrete policy/value outputs

This means transfer between PPO and IQL is technically reasonable.

### 2. Academy and `5_vs_5` should not be merged into one offline dataset by default

Even though the action and observation dimensions are compatible, they are still different tasks with different controlled-player counts and different state distributions.

That means:

- Academy and `5_vs_5` are good as sequential pretraining stages
- Academy and `5_vs_5` are bad as an untagged mixed offline dataset

### 3. Offline reward should match the intended shaped objective

If the online PPO line uses pass and shot shaping, then the offline dataset should preserve that same reward semantics when `reward_key=reward` is used.

Without that, offline RL is optimizing a different target than PPO.

## Recommended Integration Pattern

## Phase A: Academy PPO Bootstrapping

Use PPO on:

1. `academy_run_to_score_with_keeper`
2. `academy_pass_and_shoot_with_keeper`
3. `academy_3_vs_1_with_keeper`

Purpose:

- teach direct scoring
- teach pass-then-finish
- teach simple support and numerical advantage

Do not spend the majority of total compute here.

Exit condition:

- a checkpoint visibly shows passing and shot creation
- the stage is not only "completed" but behaviorally convincing

## Phase B: `5_vs_5` PPO Transfer

Take the best Academy PPO checkpoint and initialize `five_vs_five` PPO from it.

This is the current online bridge and should remain the main one.

Purpose:

- adapt Academy primitives to live `5_vs_5`
- learn spacing, transition, and more realistic decision making

This stage should still receive the largest online training budget.

## Phase C: `5_vs_5` Offline Dataset Collection

After a meaningful `5_vs_5` PPO run exists, collect offline data from `5_vs_5`, not from a mixed Academy plus `5_vs_5` soup.

Recommended sources:

- best current `5_vs_5` PPO checkpoint
- a weaker earlier `5_vs_5` checkpoint
- exploratory versions with `epsilon > 0`
- optional random data as a coverage floor

Purpose:

- preserve useful attacking behavior
- widen state-action coverage
- let offline RL revisit transitions many times

## Phase D: `5_vs_5` IQL Training

Train IQL on the `5_vs_5` offline dataset.

This is the right first offline target because:

- the final objective is `5_vs_5`
- the dataset contains live-game mistakes and opportunities
- offline updates are much cheaper than another long PPO run

Use offline RL here to improve:

- sample efficiency
- action selection from mixed-quality data
- reuse of strong transitions that PPO only saw once

## Phase E: Transfer IQL Back Into PPO

If IQL produces a cleaner attacking policy than the current PPO line, use the IQL checkpoint as the PPO initialization for another `5_vs_5` online run.

This creates the real loop:

1. PPO discovers behavior online
2. offline RL distills and improves from replay
3. PPO resumes online adaptation from the improved policy

This is the strongest combined workflow in the current codebase.

## What To Avoid

### Avoid 1: Long Academy-only offline training as the main plan

That would optimize a narrower task too strongly and risks transfer failure.

### Avoid 2: Mixing Academy and `5_vs_5` datasets in one IQL run without task identity

That is likely to create a confused offline objective.

The better pattern is:

- Academy PPO pretraining
- then `5_vs_5` PPO
- then `5_vs_5` offline RL

### Avoid 3: Judging offline RL only by reward curves

Use:

- pass completion
- shot creation
- goal difference
- video evidence

The near-term question is still whether the agent has learned real football primitives.

## Best Practical Workflow Right Now

If the goal is to make progress fast without inventing a whole new system, the best sequence is:

1. finish the Academy PPO primitive stage
2. transfer into `5_vs_5` PPO
3. collect `5_vs_5` offline data from PPO checkpoints
4. train `5_vs_5` IQL
5. compare PPO vs IQL in `5_vs_5`
6. if IQL looks better, initialize PPO from IQL and continue online

This keeps the project disciplined:

- no wasted long Academy obsession
- no blind trust in pure `5_vs_5` PPO from scratch
- no unrealistic assumption that offline RL alone will solve everything

## What The Current Code Now Supports

The current integration path should support these practical workflows:

- collect offline data from PPO checkpoints
- collect offline data from IQL checkpoints
- preserve reward-shaping configuration in collected manifests
- train IQL on a manifest-consistent dataset only
- initialize PPO directly from an IQL checkpoint

This is enough to run:

1. Academy PPO
2. `5_vs_5` PPO
3. `5_vs_5` IQL
4. PPO fine-tuning from IQL

without introducing a new algorithm family.

## Final Recommendation

For this repo, the best combined strategy is not:

- Academy instead of offline RL
- offline RL instead of PPO
- or `5_vs_5` PPO from scratch forever

The best combined strategy is:

- Academy PPO for primitives
- `5_vs_5` PPO for online transfer
- `5_vs_5` offline RL for replay-based improvement
- PPO again for final online adaptation

That is the cleanest way to make the three pieces work together.
