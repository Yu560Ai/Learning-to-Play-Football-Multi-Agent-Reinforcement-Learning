# One-Day Training Plan

## Title

One-Day Plan for Competitive Attacking Behavior in `two_v_two_plus_goalkeepers`

## Goal

Train a model that shows credible competitive attacking behavior in the custom `Two_V_Two` environment within roughly one day of compute.

Success does **not** require reliably winning or scoring.

Success **does** require visible attacking structure such as:

- advancing the ball with purpose
- avoiding immediate random turnovers
- entering dangerous attacking areas
- producing shots sometimes
- using the teammate in at least some attack sequences

## Fixed Environment

Use only the corrected custom scenario:

- scenario: `two_v_two_plus_goalkeepers`
- left side: `2` controlled outfield agents + built-in goalkeeper
- right side: `2` built-in outfield opponents + built-in goalkeeper
- full-game-style episode flow
- not academy

## Time Budget Assumption

Assume about one day total compute on the current machine.

Measured local throughput on the corrected scenario with `shared_ppo`, `4` rollout threads, `400`-step episodes is about:

- `~800` env-steps/s overall

This implies:

- `2M` steps: about `40-50` minutes
- `5M` steps: about `1.8-2.2` hours
- `10M` steps: about `4-4.5` hours

`mappo_id_cc` should be treated as slower:

- roughly `25%` to `50%` slower than `shared_ppo`

## Strategy

Because one day is limited, do **not** run the full old matrix.

Use a staged funnel:

1. cheap reward screening
2. one strong baseline run
3. one structure upgrade run if justified
4. generate videos and deterministic eval artifacts

## Priority

Prioritize:

1. getting one policy that visibly attacks
2. understanding whether teammate-aware behavior appears at all
3. only then testing a heavier multi-agent structure

Do **not** prioritize:

- full Phase 1 reward sweep at large budget
- large Phase 2 matrix
- multiple seeds on day one

## Planned Runs

### Stage A: Fast reward screen

Run shared-policy PPO only.

Conditions:

- `r2_progress + shared_ppo`
- `r3_assist + shared_ppo`

Budget:

- `2M` each

Why:

- `R2` is the stable movement/advancement baseline
- `R3` is the only reasonable cooperation-oriented candidate
- `R1` is too sparse for a one-day sprint
- `R4` is too likely to waste budget on penalty tuning

Decision after Stage A:

- if `R3` shows any pass-level or teammate-aware behavior, keep it
- otherwise use `R2` as the main day-one baseline

### Stage B: Main long run

Take the better Stage A candidate and extend it.

Budget:

- `10M`

Primary default:

- `r2_progress + shared_ppo` if `R3` shows no real behavioral value at `2M`
- `r3_assist + shared_ppo` if `R3` shows meaningful teammate-aware separation

Goal:

- produce one model with clearly non-random attacking behavior

### Stage C: Optional structure test

Only run this if time remains and Stage B produced a promising reward candidate.

Condition:

- best reward from Stage A/B + `mappo_id_cc`

Budget:

- `5M`

Purpose:

- test whether centralized critic improves attack organization or teammate usage

Skip this stage if:

- Stage B is still weak
- less than about `6` hours remain

## Evaluation Plan

Evaluation should focus on behavior, not raw return alone.

For each important checkpoint, collect:

- episode return
- goal count / goal rate
- pass count
- pass-to-shot count
- assist count
- same-owner possession length

Also inspect videos.

The checkpoint to highlight does **not** have to be `latest.pt`.

## Day-One Success Criteria

Day one is successful if at least one checkpoint shows several of these:

- repeated forward progression
- more stable possession than a random policy
- at least occasional shot creation
- visually coherent attack shape
- optional but valuable: nonzero pass count

Day one is **not** a failure if:

- goals are still rare
- assists are still zero
- deterministic win rate is still low

## Deliverables

By the end of the day, produce:

- one main trained run on the corrected custom scenario
- deterministic eval summaries for key checkpoints
- one or more rollout videos
- a short result note stating:
  - what was run
  - what behavior was observed
  - whether the next budget should go to `R2`, `R3`, or `MAPPO`

## Recommended Execution Order

1. `r2_progress + shared_ppo` at `2M`
2. `r3_assist + shared_ppo` at `2M`
3. choose one reward
4. extend chosen reward with `shared_ppo` to `10M`
5. if justified and time remains, run chosen reward with `mappo_id_cc` at `5M`
6. evaluate checkpoints and generate videos

## Practical Recommendation

For one day only, the most realistic target is:

- **one good attacking shared-PPO policy**

not:

- a full reward paper
- a full architecture paper
- a solved cooperative football agent
