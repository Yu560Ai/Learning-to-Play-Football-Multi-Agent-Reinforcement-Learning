# Arena

This folder is a head-to-head football arena parallel to `Y_Fu/` and `X_Jiang/`.

Purpose:

- run local model-vs-model matches
- keep adapter code separate from each student's training folder
- make future integrations such as `X_Jiang` plug into one stable interface

## Current Support

- `yfu_saltyfish`: load a `Y_Fu/saltyfish_baseline` checkpoint and play it on either side
- `random`: random-action baseline

There is currently no `X_Jiang` model code or checkpoint in the repository, so the arena cannot yet load a real Jiang agent. The extension point is the `ArenaAgent` interface in [agents/base.py](/home/yuhan/Codes/RL/Learning-to-Play-Football-Multi-Agent-Reinforcement-Learning/Arena/agents/base.py), and [agents/template_agent.py](/home/yuhan/Codes/RL/Learning-to-Play-Football-Multi-Agent-Reinforcement-Learning/Arena/agents/template_agent.py) shows the skeleton.

## Example Commands

Y_Fu SaltyFish vs random:

```bash
cd ~/Codes/RL/Learning-to-Play-Football-Multi-Agent-Reinforcement-Learning
source football-master/football-env/bin/activate
python -m Arena.run_match \
  --left-agent yfu_saltyfish \
  --left-checkpoint Y_Fu/checkpoints/saltyfish_baseline/latest.pt \
  --right-agent random \
  --episodes 3
```

Y_Fu SaltyFish vs Y_Fu SaltyFish:

```bash
python -m Arena.run_match \
  --left-agent yfu_saltyfish \
  --left-checkpoint Y_Fu/checkpoints/saltyfish_baseline/update_200.pt \
  --right-agent yfu_saltyfish \
  --right-checkpoint Y_Fu/checkpoints/saltyfish_baseline/update_380.pt \
  --episodes 3 \
  --deterministic-left \
  --deterministic-right
```

Fast smoke test:

```bash
python -m Arena.run_match \
  --left-agent random \
  --right-agent random \
  --episodes 1 \
  --max-steps 64
```

## How To Add X_Jiang Later

1. Create an adapter class under `Arena/agents/`, for example `x_jiang_agent.py`.
2. Implement:
   - checkpoint loading
   - raw-observation preprocessing
   - action mapping back to default football actions
3. Register it in [registry.py](/home/yuhan/Codes/RL/Learning-to-Play-Football-Multi-Agent-Reinforcement-Learning/Arena/registry.py).

Then you can run:

```bash
python -m Arena.run_match \
  --left-agent yfu_saltyfish \
  --left-checkpoint ... \
  --right-agent x_jiang \
  --right-checkpoint ...
```
