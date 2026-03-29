# Y_Fu Terminal Commands

## Start Here

From the repository root:

```bash
cd ~/Codes/RL/Learning-to-Play-Football-Multi-Agent-Reinforcement-Learning
source football-master/football-env/bin/activate
```

## Curriculum Training

Train the stages in this order:

```bash
python -u Y_Fu/train.py --preset academy_run_to_score_with_keeper --device cpu
python -u Y_Fu/train.py --preset academy_pass_and_shoot_with_keeper --device cpu
python -u Y_Fu/train.py --preset academy_3_vs_1_with_keeper --device cpu
python -u Y_Fu/train.py --preset five_vs_five --device cpu
```

## Continue Training From A Checkpoint

Continue `academy_3_vs_1_with_keeper`:

```bash
python -u Y_Fu/train.py --preset academy_3_vs_1_with_keeper --device cpu --total-timesteps 300000 --init-checkpoint Y_Fu/checkpoints/academy_3_vs_1_with_keeper/latest.pt
```

Start `academy_3_vs_1_with_keeper` from `academy_pass_and_shoot_with_keeper`:

```bash
python -u Y_Fu/train.py --preset academy_3_vs_1_with_keeper --device cpu --init-checkpoint Y_Fu/checkpoints/academy_pass_and_shoot_with_keeper/latest.pt
```

Start `five_vs_five` from `academy_3_vs_1_with_keeper`:

```bash
python -u Y_Fu/train.py --preset five_vs_five --device cpu --total-timesteps 1000000 --init-checkpoint Y_Fu/checkpoints/academy_3_vs_1_with_keeper/latest.pt
```

Start `five_vs_five` from the better `academy_3_vs_1_with_keeper` checkpoint:

```bash
python -u Y_Fu/train.py --preset five_vs_five --device cpu --total-timesteps 1000000 --init-checkpoint Y_Fu/checkpoints/academy_3_vs_1_with_keeper/update_90.pt
```

Start `five_vs_five` from the saved `five_vs_five` checkpoint with stronger offense shaping:

```bash
python -u Y_Fu/train.py --preset five_vs_five --device cpu --rollout-steps 1024 --total-timesteps 1000000 --init-checkpoint Y_Fu/checkpoints/five_vs_five/update_140.pt --pass-success-reward 0.08 --pass-failure-penalty 0.06 --pass-progress-reward-scale 0.08 --shot-attempt-reward 0.03 --attacking-possession-reward 0.0015 --final-third-entry-reward 0.04 --possession-retention-reward 0.0008 --own-half-turnover-penalty 0.04
```

Continue `five_vs_five` from the saved 5v5 checkpoint:

```bash
python -u Y_Fu/train.py --preset five_vs_five --device cpu --total-timesteps 1000000 --init-checkpoint Y_Fu/checkpoints/five_vs_five/update_70.pt
```

## Evaluation

Evaluate the current `academy_3_vs_1_with_keeper` checkpoint:

```bash
python Y_Fu/evaluate.py --checkpoint Y_Fu/checkpoints/academy_3_vs_1_with_keeper/latest.pt --episodes 10 --deterministic --compare-random --device cpu
```

Render one `academy_3_vs_1_with_keeper` episode:

```bash
python Y_Fu/evaluate.py --checkpoint Y_Fu/checkpoints/academy_3_vs_1_with_keeper/latest.pt --episodes 1 --deterministic --render --device cpu
```

Evaluate the current `five_vs_five` checkpoint:

```bash
python Y_Fu/evaluate.py --checkpoint Y_Fu/checkpoints/five_vs_five/latest.pt --episodes 5 --deterministic --compare-random --device cpu
```

Render one `five_vs_five` episode:

```bash
python Y_Fu/evaluate.py --checkpoint Y_Fu/checkpoints/five_vs_five/latest.pt --episodes 1 --deterministic --render --device cpu
```

## Save Video

Save one evaluation video:

```bash
python Y_Fu/evaluate.py --checkpoint Y_Fu/checkpoints/academy_3_vs_1_with_keeper/latest.pt --episodes 1 --deterministic --device cpu --save-video --video-dir Y_Fu/videos/academy_3_vs_1_with_keeper
```

Save video from the better `academy_3_vs_1_with_keeper` checkpoint:

```bash
python Y_Fu/evaluate.py --checkpoint Y_Fu/checkpoints/academy_3_vs_1_with_keeper/update_90.pt --episodes 20 --device cpu --save-video --video-dir Y_Fu/videos/academy_3v1_update90
```

## Notes

- Use preset-specific checkpoints, not `Y_Fu/checkpoints/latest.pt`.
- Current shared environment is `football-master/football-env`.
- Read `Y_Fu/SPEC.md` for explanations of rewards, success rate, episode endings, and PPO losses.
