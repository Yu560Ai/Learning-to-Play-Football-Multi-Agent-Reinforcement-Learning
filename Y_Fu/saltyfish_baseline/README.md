# SaltyFish-Inspired Baseline

This folder contains a separate baseline inspired by the public summary of the Kaggle Google Football `SaltyFish` solution.

What is reproduced locally:

- single-player competition-style environment: `11_vs_11_kaggle`
- `simple115v2` observation
- grouped feature heads instead of one flat encoder
- reduced static action set
- plain PPO training on a single machine

What is not reproduced exactly:

- distributed IMPALA training
- full league self-play
- behavior cloning from competition trajectories
- Kaggle ladder-specific evaluation

Use this as a more systematic competition-style baseline, not as a claim of exact reproduction.

Train:

```bash
python Y_Fu/train_saltyfish.py --device cpu
```

Evaluate:

```bash
python Y_Fu/evaluate_saltyfish.py --checkpoint Y_Fu/checkpoints/saltyfish_baseline/latest.pt --episodes 5 --compare-random --device cpu
```
