# Environment Reset

## Intended task

`Two_V_Two` is now reset around the intended custom GRF scenario:

- `2` controlled left-side outfield agents
- built-in left goalkeeper
- `2` built-in right-side outfield opponents
- built-in right goalkeeper
- full-game-style episode flow
- shared cooperative reward over the two controlled agents

Scenario name:

- `two_v_two_plus_goalkeepers`

## What was wrong before

Earlier Codex-written docs and defaults drifted to the built-in academy baseline:

- `academy_run_pass_and_shoot_with_keeper`

That academy setup is not the intended research environment for this project.

## How to treat old artifacts

Existing results produced under the academy baseline should be treated as:

- legacy infrastructure checks
- legacy reward/algorithm debugging artifacts
- not valid scientific evidence for the intended `Two_V_Two` task

This applies to:

- prior Phase 1 reward runs
- prior Phase 2 structure runs
- previously generated evaluation videos from those checkpoints

## What remains reusable

The following code remains reusable after the reset:

- PPO / MAPPO training backbone
- reward shaping implementation
- structure variants
- deterministic evaluation scripts
- checkpoint sweep tooling
- video rendering scripts

## Restart rule

From this point on, new experiments should use the custom scenario and should not rely on academy-based checkpoints or summaries.
