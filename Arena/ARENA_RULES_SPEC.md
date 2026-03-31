# Arena Rules Spec

## Purpose

This file defines the team-wide rule for using `Arena/` to evaluate shared football agents.

The main goal is to make Arena evaluation:

- consistent across teammates
- reproducible
- aligned with the final project scope
- safe for Git history and shared model management

## Main Rule

Arena should evaluate only curated shared models from `best_models/<owner>/`.

Arena should not be used directly on large personal checkpoint trees such as:

- `Y_Fu/checkpoints/...`
- `X_Jiang/checkpoints/...`
- other local training dump folders

Those personal checkpoint trees are for local training and diagnosis only.

## Shared Model Eligibility

A model is eligible for Arena only if all of the following are true:

1. It is copied into the owner's shared folder under `best_models/<owner>/`.
2. The owner folder includes `MODEL_SPEC.txt`.
3. The model is intended as a curated candidate, not a random intermediate checkpoint.
4. The model has an Arena adapter that can load it correctly.

If any of these conditions fail, the model is not part of the official Arena comparison.

## Owner Folder Rule

Use this structure:

- `best_models/Y_Fu/`
- `best_models/Y_Yao/`
- `best_models/X_Jiang/`

Each owner folder may contain:

- one or two curated `.pt` model files at most
- one `MODEL_SPEC.txt` file describing the current shared model choice

If an owner folder has no real model file yet, that owner's Arena matches are blocked and should be reported as unavailable rather than skipped silently.

Current team expectation:

- `Y_Fu`, `Y_Yao`, and `X_Jiang` should each place one curated shared model in their own `best_models/<owner>/` folder.
- Once all three shared model files are present, the full Arena evaluation matrix should be run.

## Final Project Scope Rule

For the final two weeks, Arena should focus on the same main task as the project:

- `five_vs_five`

This means:

- `five_vs_five` is the primary Arena evaluation target
- curriculum stages such as `2_agents` and `3_agents` are secondary evidence only
- `11v11` is not the main Arena target unless extra time remains

## Official Arena Evaluation Protocol

For the current shared multi-agent line, use:

- environment: `5_vs_5`
- representation: `extracted`
- action set: `v2`
- left controlled players: `4`
- right controlled players: `4`
- channel size: `42 x 42`

This matches the current shared `Y_Fu` five-vs-five checkpoint family.

## Baseline Rule

The official baseline is the Google built-in policy, exposed in Arena as:

- `google_builtin`

Every shared model should be evaluated against the Google built-in baseline in both directions:

1. shared model on the left vs Google built-in on the right
2. Google built-in on the left vs shared model on the right

This avoids side bias and gives a cleaner comparison.

## Qualification Gate

Before a shared model is allowed to enter official inter-owner matches, it must first pass the Google-baseline gate.

The rule is:

- if the shared model loses to `google_builtin` overall, it is not qualified for owner-vs-owner Arena matches
- if the shared model is at least competitive with `google_builtin`, it may enter the inter-owner table

For the default workflow, "loses overall" means the combined two-direction evaluation against `google_builtin` shows a negative result for the shared model. The main reference fields are:

- win/loss record
- `avg_goal_diff`
- `avg_score`

If the shared model clearly finishes behind the Google baseline, stop there and report it as:

- baseline failed
- not eligible for inter-owner Arena play

Do not spend Arena time on owner-vs-owner matches for a model that cannot clear the Google-baseline gate.

## Inter-Owner Match Rule

If all three owners have valid shared models and they pass the Google-baseline gate, run pairwise inter-owner matches:

1. `Y_Fu` vs `Y_Yao`
2. `Y_Fu` vs `X_Jiang`
3. `Y_Yao` vs `X_Jiang`

For each pairing, run both directions:

1. owner A on the left, owner B on the right
2. owner B on the left, owner A on the right

If a model is missing for one owner, record the pairing as blocked.

If a model fails the Google-baseline gate, record that owner's inter-owner pairings as blocked by qualification.

Do not fabricate results and do not substitute random or placeholder policies for a missing owner model in the official inter-owner table.

## Default Match Count

Because time is limited, use this default:

- quick comparison: `3` episodes per direction
- stronger comparison: `10` episodes per direction

For the current final-stage workflow, `3` episodes per direction is acceptable for quick Arena checks.

If a result is important enough to quote in a report, rerun with a larger sample.

## Reporting Rule

Every Arena evaluation report should include:

1. which model files were actually present
2. which pairings were run
3. which pairings were blocked
4. which owners passed or failed the Google-baseline gate
5. the exact Arena command used
6. summary metrics from `arena_summary`

Required summary fields:

- `left_avg_reward`
- `right_avg_reward`
- `avg_score`
- `avg_goal_diff`
- `left_win_rate`
- `draw_rate`
- `avg_length`

When two directions are run, include a short combined interpretation.

## Git Rule

Arena evaluation should use only committed shared models from `best_models/` or clearly identified local shared candidates in that same folder.

Do not commit:

- personal checkpoint trees
- replay dumps
- evaluation videos
- temporary local Arena artifacts

Arena reports may be committed if they are short, text-based, and useful to the team.

## Current Practical Status

At the moment, the Arena structure is prepared for all three teammates:

- `best_models/Y_Fu/`
- `best_models/Y_Yao/`
- `best_models/X_Jiang/`

The intended steady-state workflow is:

- each teammate keeps one curated shared model in their own folder
- Arena runs the Google-baseline comparison for each owner first
- only owners that pass the Google-baseline gate move on to inter-owner evaluation
- Arena runs the full inter-owner round-robin only after all required model files are available and the relevant models pass qualification

## Decision Rule

If Arena work conflicts with core model-improvement time, prioritize model improvement.

Arena is a comparison tool, not the main training objective.

For the final project phase:

- first priority: improve the shared `five_vs_five` model
- second priority: compare shared models in Arena
- third priority: extend Arena only when it helps answer the main project question
