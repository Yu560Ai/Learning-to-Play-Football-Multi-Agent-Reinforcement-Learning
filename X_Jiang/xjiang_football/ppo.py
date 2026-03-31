from __future__ import annotations

import math
import random
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch import nn

from presets import TrainConfig, build_config
from xjiang_football.envs import FootballEnvWrapper
from xjiang_football.model import ActorCritic


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_gae(
    rewards: np.ndarray,
    values: np.ndarray,
    dones: np.ndarray,
    last_values: np.ndarray,
    gamma: float,
    gae_lambda: float,
) -> tuple[np.ndarray, np.ndarray]:
    advantages = np.zeros_like(rewards, dtype=np.float32)
    last_advantage = np.zeros(rewards.shape[1], dtype=np.float32)

    for step in reversed(range(rewards.shape[0])):
        if step == rewards.shape[0] - 1:
            next_values = last_values
        else:
            next_values = values[step + 1]

        next_non_terminal = 1.0 - dones[step]
        delta = rewards[step] + gamma * next_values * next_non_terminal - values[step]
        last_advantage = delta + gamma * gae_lambda * next_non_terminal * last_advantage
        advantages[step] = last_advantage

    returns = advantages + values
    return advantages, returns


class PPOTrainer:
    def __init__(self, config: TrainConfig) -> None:
        self.config = config
        self.device = resolve_device(config.device)
        set_seed(config.seed)

        self.env = FootballEnvWrapper(
            env_name=config.env_name,
            representation=config.representation,
            rewards=config.rewards,
            render=config.render,
            logdir=config.logdir,
            num_controlled_players=config.num_controlled_players,
            channel_dimensions=config.channel_dimensions,
        )

        self.model = ActorCritic(
            obs_dim=self.env.obs_dim,
            action_dim=self.env.action_dim,
            hidden_sizes=config.hidden_sizes,
            obs_shape=self.env.obs_shape,
            model_type=config.model_type,
            feature_dim=config.feature_dim,
        ).to(self.device)

        if config.init_checkpoint is not None:
            self.load_initial_checkpoint(config.init_checkpoint)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.learning_rate, eps=1e-5)

        self.save_dir = Path(config.save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        Path(config.logdir).mkdir(parents=True, exist_ok=True)

        self.total_agent_steps = 0
        self.current_episode_return = 0.0
        self.current_episode_score_reward = 0.0
        self.current_episode_length = 0

    def load_initial_checkpoint(self, checkpoint_path: str) -> None:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        self.model.load_state_dict(checkpoint["model_state_dict"], strict=True)

    def collect_rollout(
        self,
        observation: np.ndarray,
    ) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, float]]:
        steps = self.config.rollout_steps
        num_players = self.env.num_players

        obs_buffer = np.zeros((steps, num_players, self.env.obs_dim), dtype=np.float32)
        action_buffer = np.zeros((steps, num_players), dtype=np.int64)
        logprob_buffer = np.zeros((steps, num_players), dtype=np.float32)
        reward_buffer = np.zeros((steps, num_players), dtype=np.float32)
        done_buffer = np.zeros(steps, dtype=np.float32)
        value_buffer = np.zeros((steps, num_players), dtype=np.float32)

        completed_returns: list[float] = []
        completed_lengths: list[int] = []
        completed_score_rewards: list[float] = []
        completed_successes: list[float] = []
        completed_scores: list[tuple[int, int]] = []

        for step in range(steps):
            obs_buffer[step] = observation

            obs_tensor = torch.as_tensor(observation, dtype=torch.float32, device=self.device)
            with torch.no_grad():
                actions, logprobs, values = self.model.act(obs_tensor)

            action_np = actions.cpu().numpy()
            next_observation, reward, done, info = self.env.step(action_np)

            action_buffer[step] = action_np
            logprob_buffer[step] = logprobs.cpu().numpy()
            value_buffer[step] = values.cpu().numpy()
            reward_buffer[step] = reward
            done_buffer[step] = float(done)

            self.current_episode_return += float(np.mean(reward))
            self.current_episode_score_reward += float(info.get("score_reward", 0.0))
            self.current_episode_length += 1
            self.total_agent_steps += num_players

            observation = next_observation

            if done:
                score = self.env.get_score()
                if score is not None:
                    success = 1.0 if score[0] > score[1] else 0.0
                    completed_scores.append((int(score[0]), int(score[1])))
                else:
                    success = 1.0 if self.current_episode_score_reward > 0.0 else 0.0

                completed_returns.append(self.current_episode_return)
                completed_lengths.append(self.current_episode_length)
                completed_score_rewards.append(self.current_episode_score_reward)
                completed_successes.append(success)

                self.current_episode_return = 0.0
                self.current_episode_score_reward = 0.0
                self.current_episode_length = 0

                observation = self.env.reset()

        with torch.no_grad():
            last_values = self.model.get_value(
                torch.as_tensor(observation, dtype=torch.float32, device=self.device)
            ).cpu().numpy()

        advantages, returns = compute_gae(
            rewards=reward_buffer,
            values=value_buffer,
            dones=done_buffer,
            last_values=last_values,
            gamma=self.config.gamma,
            gae_lambda=self.config.gae_lambda,
        )

        batch = {
            "obs": obs_buffer.reshape(-1, self.env.obs_dim),
            "actions": action_buffer.reshape(-1),
            "logprobs": logprob_buffer.reshape(-1),
            "advantages": advantages.reshape(-1),
            "returns": returns.reshape(-1),
            "values": value_buffer.reshape(-1),
        }

        score_examples = (
            ",".join(f"{left}-{right}" for left, right in completed_scores[-3:])
            if completed_scores
            else "n/a"
        )

        metrics = {
            "episodes_finished": float(len(completed_returns)),
            "mean_episode_return": float(np.mean(completed_returns)) if completed_returns else float("nan"),
            "mean_episode_length": float(np.mean(completed_lengths)) if completed_lengths else float("nan"),
            "mean_score_reward": float(np.mean(completed_score_rewards)) if completed_score_rewards else float("nan"),
            "mean_goals_for": float(np.mean([left for left, _ in completed_scores])) if completed_scores else float("nan"),
            "mean_goals_against": float(np.mean([right for _, right in completed_scores])) if completed_scores else float("nan"),
            "score_examples": score_examples,
            "success_rate": float(np.mean(completed_successes)) if completed_successes else float("nan"),
        }

        return observation, batch, metrics

    def update(self, batch: dict[str, np.ndarray]) -> dict[str, float]:
        obs = torch.as_tensor(batch["obs"], dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(batch["actions"], dtype=torch.int64, device=self.device)
        old_logprobs = torch.as_tensor(batch["logprobs"], dtype=torch.float32, device=self.device)
        advantages = torch.as_tensor(batch["advantages"], dtype=torch.float32, device=self.device)
        returns = torch.as_tensor(batch["returns"], dtype=torch.float32, device=self.device)
        old_values = torch.as_tensor(batch["values"], dtype=torch.float32, device=self.device)

        if advantages.numel() > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)

        batch_size = obs.shape[0]
        minibatch_size = batch_size // self.config.num_minibatches
        if minibatch_size < 1:
            raise ValueError("num_minibatches is too large for the collected rollout.")

        policy_losses: list[float] = []
        value_losses: list[float] = []
        entropies: list[float] = []
        approx_kls: list[float] = []
        clipfracs: list[float] = []

        for _ in range(self.config.update_epochs):
            indices = np.random.permutation(batch_size)

            for start in range(0, batch_size, minibatch_size):
                mb_idx = indices[start : start + minibatch_size]

                _, new_logprob, entropy, new_value = self.model.get_action_and_value(
                    obs[mb_idx],
                    actions[mb_idx],
                )

                logratio = new_logprob - old_logprobs[mb_idx]
                ratio = logratio.exp()

                with torch.no_grad():
                    approx_kl = ((ratio - 1.0) - logratio).mean()
                    clipfracs.append(float(((ratio - 1.0).abs() > self.config.clip_coef).float().mean().item()))

                mb_advantages = advantages[mb_idx]

                policy_loss_1 = -mb_advantages * ratio
                policy_loss_2 = -mb_advantages * torch.clamp(
                    ratio,
                    1.0 - self.config.clip_coef,
                    1.0 + self.config.clip_coef,
                )
                policy_loss = torch.max(policy_loss_1, policy_loss_2).mean()

                value_delta = new_value - old_values[mb_idx]
                value_clipped = old_values[mb_idx] + torch.clamp(
                    value_delta,
                    -self.config.clip_coef,
                    self.config.clip_coef,
                )

                value_loss_unclipped = (new_value - returns[mb_idx]) ** 2
                value_loss_clipped = (value_clipped - returns[mb_idx]) ** 2
                value_loss = 0.5 * torch.max(value_loss_unclipped, value_loss_clipped).mean()

                entropy_loss = entropy.mean()
                loss = policy_loss + self.config.vf_coef * value_loss - self.config.ent_coef * entropy_loss

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                self.optimizer.step()

                policy_losses.append(float(policy_loss.item()))
                value_losses.append(float(value_loss.item()))
                entropies.append(float(entropy_loss.item()))
                approx_kls.append(float(approx_kl.item()))

        return {
            "policy_loss": float(np.mean(policy_losses)),
            "value_loss": float(np.mean(value_losses)),
            "entropy": float(np.mean(entropies)),
            "approx_kl": float(np.mean(approx_kls)),
            "clipfrac": float(np.mean(clipfracs)) if clipfracs else 0.0,
        }

    def save_checkpoint(self, name: str, update: int) -> Path:
        checkpoint_path = self.save_dir / name
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "config": asdict(self.config),
                "obs_dim": self.env.obs_dim,
                "obs_shape": self.env.obs_shape,
                "action_dim": self.env.action_dim,
                "num_players": self.env.num_players,
                "update": update,
                "total_agent_steps": self.total_agent_steps,
            },
            checkpoint_path,
        )
        return checkpoint_path

    def train(self) -> Path:
        steps_per_update = self.config.rollout_steps * self.env.num_players
        num_updates = math.ceil(self.config.total_timesteps / steps_per_update)

        observation = self.env.reset()
        latest_checkpoint = self.save_dir / "latest.pt"

        for update in range(1, num_updates + 1):
            observation, batch, rollout_metrics = self.collect_rollout(observation)
            update_metrics = self.update(batch)

            if update % self.config.log_interval == 0:
                print(
                    f"[update {update}/{num_updates}] "
                    f"agent_steps={self.total_agent_steps} "
                    f"policy_loss={update_metrics['policy_loss']:.4f} "
                    f"value_loss={update_metrics['value_loss']:.4f} "
                    f"entropy={update_metrics['entropy']:.4f} "
                    f"approx_kl={update_metrics['approx_kl']:.5f} "
                    f"episodes_finished={int(rollout_metrics['episodes_finished'])} "
                    f"episode_return={rollout_metrics['mean_episode_return']:.3f} "
                    f"score_reward={rollout_metrics['mean_score_reward']:.3f} "
                    f"goals_for={rollout_metrics['mean_goals_for']:.2f} "
                    f"goals_against={rollout_metrics['mean_goals_against']:.2f} "
                    f"score_examples={rollout_metrics['score_examples']} "
                    f"episode_length={rollout_metrics['mean_episode_length']:.1f} "
                    f"success_rate={rollout_metrics['success_rate']:.3f}"
                )

            if update % self.config.save_interval == 0:
                latest_checkpoint = self.save_checkpoint(f"update_{update}.pt", update)

        latest_checkpoint = self.save_checkpoint("latest.pt", num_updates)
        self.env.close()
        print(f"Training finished. Latest checkpoint: {latest_checkpoint}")
        return latest_checkpoint


def main(preset_name: str = "five_v_five_debug", device_override: str | None = None) -> Path:
    overrides = {}
    if device_override is not None:
        overrides["device"] = device_override

    config = build_config(preset_name, overrides=overrides)
    trainer = PPOTrainer(config)
    return trainer.train()