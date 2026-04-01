# Y_Fu Research Todo Spec

## Purpose

This file defines how to use outside study material to improve the `Y_Fu/` training line.

The goal is not to review general RL theory.

The goal is to extract training strategies that can realistically improve the current multi-agent football results within the remaining project time.

## Time Constraint Rule

Only about two weeks remain.

So reading must stay narrow and practical.

Do not spend time on broad background reading unless it directly changes:

- training stability
- transfer quality
- evaluation quality
- final `five_vs_five` performance

## Main Focus Topics

When reading slides, textbooks, videos, or papers, focus on these topics:

- curriculum transfer: `2 -> 3 -> 5`
- shared-policy PPO stability
- CTDE / MAPPO ideas
- reward shaping for sparse football rewards
- checkpoint selection and evaluation protocol
- self-play or stronger baselines
- multi-agent credit assignment

## Reading Workflow

Use this workflow:

1. Provide the material first.
2. Extract only training-relevant ideas.
3. Map each idea to a concrete action for `Y_Fu/`.
4. Rank actions by usefulness and implementation cost.
5. Turn the reading into an experiment plan.

Do not stop at summarization.

The reading should end in an actionable training decision.

## What To Provide

The material can be any of:

- local slide files
- textbook chapters or PDFs
- YouTube teaching links
- research paper links
- research paper PDFs

If there are many materials, send the most important `3` to `8` first.

## Required Inputs

To process the materials efficiently, provide:

- local file paths for slides or textbook PDFs
- YouTube links
- paper links or local PDFs

## Expected Output

After reviewing the materials, produce files such as:

- `Y_Fu/TRAINING_STRATEGY_NOTES.md`
- `Y_Fu/NEXT_EXPERIMENTS.md`

These outputs should contain:

- the key idea from each source
- whether it is useful for the current `Y_Fu` football setup
- the exact code or training change to try
- the priority level: do now, later, or skip

## Decision Standard

A reading insight is worth keeping only if it helps answer at least one of these:

- how to make the `2 -> 3 -> 5` curriculum transfer more effective
- how to make shared-policy PPO more stable
- how to improve `five_vs_five` outcome metrics
- how to evaluate checkpoints more reliably
- how to beat or at least close the gap to the Google baseline

If an idea does not affect one of those, it is lower priority.

## Current Context

Current project direction:

- main target: `five_vs_five`
- current training curriculum: `academy_pass_and_shoot_with_keeper -> academy_3_vs_1_with_keeper -> five_vs_five`
- current Arena result: the shared `Y_Fu` `five_vs_five` model is still below the Google built-in baseline

So outside reading should be used to improve that exact line.
