from __future__ import annotations

import argparse
import math
import random
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from .envs import FootballEnvWrapper
from .model import ActorCritic


@dataclass
class PPOConfig:
    env_name: str = "11_vs_11_easy_stochastic"
    representation: str = "simple115v2"
    rewards: str = "scoring,checkpoints"
    num_controlled_players: int = 11
    channel_dimensions: tuple[int, int] = (42, 42)
    total_timesteps: int = 500_000
    rollout_steps: int = 256
    update_epochs: int = 4
    num_minibatches: int = 8
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    hidden_sizes: tuple[int, ...] = (256, 256)
    seed: int = 42
    render: bool = False
    save_interval: int = 10
    log_interval: int = 1
    save_dir: str = "Y_Fu/checkpoints"
    logdir: str = "Y_Fu/logs"
    device: str = "auto"


PRESET_OVERRIDES: dict[str, dict[str, Any]] = {
    "default": {},
    "lightning": {
        "env_name": "academy_empty_goal_close",
        "representation": "extracted",
        "rewards": "scoring,checkpoints",
        "num_controlled_players": 1,
        "channel_dimensions": (42, 42),
        "total_timesteps": 20_000,
        "rollout_steps": 128,
        "update_epochs": 3,
        "num_minibatches": 4,
        "learning_rate": 3e-4,
        "hidden_sizes": (128, 128),
        "save_interval": 2,
    },
    "small_11v11": {
        "env_name": "11_vs_11_easy_stochastic",
        "representation": "simple115v2",
        "rewards": "scoring,checkpoints",
        "num_controlled_players": 3,
        "channel_dimensions": (42, 42),
        "total_timesteps": 60_000,
        "rollout_steps": 128,
        "update_epochs": 3,
        "num_minibatches": 4,
        "hidden_sizes": (128, 128),
        "save_interval": 5,
    },
}


def _resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _compute_gae(
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
    def __init__(self, config: PPOConfig) -> None:
        self.config = config
        self.device = _resolve_device(config.device)
        _set_seed(config.seed)

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
        ).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.learning_rate, eps=1e-5)
        self.save_dir = Path(config.save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        Path(config.logdir).mkdir(parents=True, exist_ok=True)

        self.total_agent_steps = 0
        self.current_episode_return = 0.0
        self.current_episode_length = 0

    def collect_rollout(self, observation: np.ndarray) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, float]]:
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

        for step in range(steps):
            obs_buffer[step] = observation
            obs_tensor = torch.as_tensor(observation, dtype=torch.float32, device=self.device)
            with torch.no_grad():
                actions, logprobs, values = self.model.act(obs_tensor)

            action_np = actions.cpu().numpy()
            next_observation, reward, done, _ = self.env.step(action_np)

            action_buffer[step] = action_np
            logprob_buffer[step] = logprobs.cpu().numpy()
            value_buffer[step] = values.cpu().numpy()
            reward_buffer[step] = reward
            done_buffer[step] = float(done)

            self.current_episode_return += float(np.mean(reward))
            self.current_episode_length += 1
            self.total_agent_steps += num_players

            observation = next_observation
            if done:
                completed_returns.append(self.current_episode_return)
                completed_lengths.append(self.current_episode_length)
                self.current_episode_return = 0.0
                self.current_episode_length = 0
                observation = self.env.reset()

        with torch.no_grad():
            last_values = self.model.get_value(
                torch.as_tensor(observation, dtype=torch.float32, device=self.device)
            ).cpu().numpy()

        advantages, returns = _compute_gae(
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
        metrics = {
            "episodes_finished": float(len(completed_returns)),
            "mean_episode_return": float(np.mean(completed_returns)) if completed_returns else float("nan"),
            "mean_episode_length": float(np.mean(completed_lengths)) if completed_lengths else float("nan"),
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
                minibatch_indices = indices[start : start + minibatch_size]
                _, new_logprob, entropy, new_value = self.model.get_action_and_value(
                    obs[minibatch_indices],
                    actions[minibatch_indices],
                )
                logratio = new_logprob - old_logprobs[minibatch_indices]
                ratio = logratio.exp()

                with torch.no_grad():
                    approx_kl = ((ratio - 1.0) - logratio).mean()
                    clipfracs.append(
                        float(((ratio - 1.0).abs() > self.config.clip_coef).float().mean().item())
                    )

                minibatch_advantages = advantages[minibatch_indices]
                policy_loss_1 = -minibatch_advantages * ratio
                policy_loss_2 = -minibatch_advantages * torch.clamp(
                    ratio,
                    1.0 - self.config.clip_coef,
                    1.0 + self.config.clip_coef,
                )
                policy_loss = torch.max(policy_loss_1, policy_loss_2).mean()

                value_delta = new_value - old_values[minibatch_indices]
                value_clipped = old_values[minibatch_indices] + torch.clamp(
                    value_delta,
                    -self.config.clip_coef,
                    self.config.clip_coef,
                )
                value_loss_unclipped = (new_value - returns[minibatch_indices]) ** 2
                value_loss_clipped = (value_clipped - returns[minibatch_indices]) ** 2
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
                    f"episode_return={rollout_metrics['mean_episode_return']:.3f} "
                    f"episode_length={rollout_metrics['mean_episode_length']:.1f}"
                )

            if update % self.config.save_interval == 0:
                latest_checkpoint = self.save_checkpoint(f"update_{update}.pt", update)

        latest_checkpoint = self.save_checkpoint("latest.pt", num_updates)
        self.env.close()
        print(f"Training finished. Latest checkpoint: {latest_checkpoint}")
        return latest_checkpoint


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a shared-policy PPO agent on Google Research Football.")
    parser.add_argument("--preset", choices=sorted(PRESET_OVERRIDES.keys()), default="default")
    parser.add_argument("--env-name", default="11_vs_11_easy_stochastic")
    parser.add_argument("--representation", default="simple115v2")
    parser.add_argument("--rewards", default="scoring,checkpoints")
    parser.add_argument("--num-controlled-players", type=int, default=11)
    parser.add_argument("--channel-width", type=int, default=42)
    parser.add_argument("--channel-height", type=int, default=42)
    parser.add_argument("--total-timesteps", type=int, default=500_000)
    parser.add_argument("--rollout-steps", type=int, default=256)
    parser.add_argument("--update-epochs", type=int, default=4)
    parser.add_argument("--num-minibatches", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-coef", type=float, default=0.2)
    parser.add_argument("--ent-coef", type=float, default=0.01)
    parser.add_argument("--vf-coef", type=float, default=0.5)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--hidden-sizes", type=int, nargs="+", default=[256, 256])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--save-interval", type=int, default=10)
    parser.add_argument("--log-interval", type=int, default=1)
    parser.add_argument("--save-dir", default="Y_Fu/checkpoints")
    parser.add_argument("--logdir", default="Y_Fu/logs")
    parser.add_argument("--device", default="auto")
    return parser


def _build_default_arg_values() -> dict[str, Any]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--preset", choices=sorted(PRESET_OVERRIDES.keys()), default="default")
    parser.add_argument("--env-name", default="11_vs_11_easy_stochastic")
    parser.add_argument("--representation", default="simple115v2")
    parser.add_argument("--rewards", default="scoring,checkpoints")
    parser.add_argument("--num-controlled-players", type=int, default=11)
    parser.add_argument("--channel-width", type=int, default=42)
    parser.add_argument("--channel-height", type=int, default=42)
    parser.add_argument("--total-timesteps", type=int, default=500_000)
    parser.add_argument("--rollout-steps", type=int, default=256)
    parser.add_argument("--update-epochs", type=int, default=4)
    parser.add_argument("--num-minibatches", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-coef", type=float, default=0.2)
    parser.add_argument("--ent-coef", type=float, default=0.01)
    parser.add_argument("--vf-coef", type=float, default=0.5)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--hidden-sizes", type=int, nargs="+", default=[256, 256])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--save-interval", type=int, default=10)
    parser.add_argument("--log-interval", type=int, default=1)
    parser.add_argument("--save-dir", default="Y_Fu/checkpoints")
    parser.add_argument("--logdir", default="Y_Fu/logs")
    parser.add_argument("--device", default="auto")
    return vars(parser.parse_args([]))


def config_from_args(args: argparse.Namespace) -> PPOConfig:
    defaults = _build_default_arg_values()
    merged: dict[str, Any] = dict(PRESET_OVERRIDES[args.preset])

    for key, value in vars(args).items():
        if key == "preset":
            continue
        if key not in merged or value != defaults[key]:
            merged[key] = value

    merged["channel_dimensions"] = (merged.pop("channel_width"), merged.pop("channel_height"))
    merged["hidden_sizes"] = tuple(merged["hidden_sizes"])
    merged["num_controlled_players"] = merged["num_controlled_players"]
    return PPOConfig(**merged)


def main(argv: Sequence[str] | None = None) -> Path:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    trainer = PPOTrainer(config_from_args(args))
    return trainer.train()


if __name__ == "__main__":
    main()
