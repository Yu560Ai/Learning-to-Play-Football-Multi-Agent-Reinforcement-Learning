# Learning Football Strategies with Multi-Agent Reinforcement Learning

## Aim

The aim of this project is to study how reinforcement learning (RL) agents can learn to play football in a simulated environment. In particular, we focus on **multi-agent reinforcement learning**, where multiple agents must cooperate with teammates and compete against opponents.

The project investigates whether modern RL methods can learn effective football strategies such as passing, positioning, defending, and shooting through interaction with the environment.

---
## Problem Formulation

This project studies reinforcement learning agents in the **Google Research Football** simulation environment.

The goal is to train an agent that can interact with the football simulator and learn strategies through reinforcement learning.

Main resources:

- Kaggle Competition (problem description and evaluation setup)  
  https://www.kaggle.com/competitions/google-football  

- Google Research Football Environment (official code and simulator)  
  https://github.com/google-research/football
---

## What We Plan To Do

1. Train reinforcement learning agents in the football environment.
2. Implement and compare several RL algorithms.
3. Allow different trained agents to compete against each other.
4. Evaluate performance using win rate and other metrics.
5. Analyze whether meaningful strategies emerge during training.

A possible team structure:

- Member A: PPO baseline
- Member B: Multi-agent PPO (MAPPO)
- Member C: alternative RL method or improved reward shaping

Agents will be evaluated by playing against:

- built-in AI opponents
- other trained agents
- earlier training checkpoints

---

## Environment and Resources

### Football Simulation Environment

We will use the **Google Research Football environment**.

Main features:

- realistic football simulation
- configurable scenarios (1v1, 3v3, full matches)
- single-agent and multi-agent modes
- feature-based or pixel-based observations

Repository:

https://github.com/google-research/football

---

### Computing Resources

Training reinforcement learning agents requires significant computation.

Possible resources:

- local GPU machines
- university GPU servers
- parallel simulation environments

Training may require millions of environment steps.

---

## Candidate Algorithms

The following reinforcement learning algorithms will be considered.

### PPO (Proximal Policy Optimization)

A policy gradient algorithm that performs stable policy updates using a clipped objective function.

Widely used in many RL environments.

---

### MAPPO (Multi-Agent PPO)

An extension of PPO designed for multi-agent environments.

Allows multiple agents to learn coordinated behaviors.

---

### DQN (Deep Q-Network)

A value-based reinforcement learning method used as a baseline for comparison.

---

### Self-Play Training

Agents improve by playing against copies of themselves or previously trained versions.

This technique has been used in major game AI systems.

---

## Evaluation Metrics

Performance will be evaluated using:

- win rate
- goal difference
- average reward
- training stability

We will also qualitatively observe whether agents develop strategies such as passing or defensive positioning.

---

## Key Papers and References

### Google Research Football

Kurach et al., 2020  
"Google Research Football: A Novel Reinforcement Learning Environment"

https://arxiv.org/abs/1907.11180

---

### PPO

Schulman et al., 2017  
"Proximal Policy Optimization Algorithms"

https://arxiv.org/abs/1707.06347

---

### MAPPO

Yu et al., 2021  
"The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games"

https://arxiv.org/abs/2103.01955

---

### Deep Q-Network

Mnih et al., 2015  
"Human-level control through deep reinforcement learning"

https://www.nature.com/articles/nature14236

---

### Self-Play Reinforcement Learning

Silver et al., 2018  
"A General Reinforcement Learning Algorithm that Masters Chess, Shogi and Go through Self-Play"

https://science.org/doi/10.1126/science.aar6404
