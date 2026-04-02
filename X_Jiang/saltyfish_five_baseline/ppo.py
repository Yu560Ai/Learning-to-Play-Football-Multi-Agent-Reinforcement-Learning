from __future__ import annotations

import argparse
import math
import random
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .env import CompetitionEnvConfig, CompetitionRewardConfig, ReducedActionFootballEnv
from .model import SaltyFishModelConfig, StructuredSimple115ActorCritic


@dataclass
class SaltyFishPPOConfig:
    env_name: str = "5_vs_5"
    representation: str = "simple115v2"
    rewards: str = "scoring"
    num_controlled_players: int = 4
    total_timesteps: int = 204_800
    rollout_steps: int = 1024
    update_epochs: int = 4
    num_minibatches: int = 8
    learning_rate: float = 1e-4
    gamma: float = 0.993
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    head_dim: int = 64
    trunk_dim: int = 256
    seed: int = 42
    render: bool = False
    save_interval: int = 10
    log_interval: int = 1
    save_dir: str = "X_Jiang/checkpoints/saltyfish_five_v_five"
    logdir: str = "X_Jiang/logs/saltyfish_five_v_five"
    device: str = "auto"
    init_checkpoint: str | None = None
    use_engineered_features: bool = True
    possession_gain_reward: float = 0.2
    possession_loss_penalty: float = 0.2
    team_possession_reward: float = 0.001
    opponent_possession_penalty: float = 0.001
    successful_pass_reward: float = 0.02
    progressive_pass_reward_scale: float = 0.05
    carry_progress_reward_scale: float = 0.04
    attacking_third_reward: float = 0.002
    shots_with_ball_reward: float = 0.01
    attacking_risk_x_threshold: float = 0.4
    attacking_loss_penalty_scale: float = 0.5
    out_of_play_loss_penalty: float = 0.05


def _resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


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
        next_values = last_values if step == rewards.shape[0] - 1 else values[step + 1]
        next_non_terminal = 1.0 - dones[step]
        delta = rewards[step] + gamma * next_values * next_non_terminal - values[step]
        last_advantage = delta + gamma * gae_lambda * next_non_terminal * last_advantage
        advantages[step] = last_advantage
    returns = advantages + values
    return advantages, returns


def _load_compatible_state_dict(model: nn.Module, state_dict: dict[str, torch.Tensor]) -> tuple[int, int]:
    model_state = model.state_dict()
    compatible_state = {}
    skipped = 0
    for key, value in state_dict.items():
        if key in model_state and model_state[key].shape == value.shape:
            compatible_state[key] = value
        else:
            skipped += 1
    model.load_state_dict(compatible_state, strict=False)
    return len(compatible_state), skipped


class SaltyFishPPOTrainer:
    def __init__(self, config: SaltyFishPPOConfig) -> None:
        self.config = config
        self.device = _resolve_device(config.device)
        _set_seed(config.seed)

        env_config = CompetitionEnvConfig(
            env_name=config.env_name,
            representation=config.representation,
            rewards=config.rewards,
            num_controlled_players=config.num_controlled_players,
            use_engineered_features=config.use_engineered_features,
            reward_config=CompetitionRewardConfig(
                possession_gain_reward=config.possession_gain_reward,
                possession_loss_penalty=config.possession_loss_penalty,
                team_possession_reward=config.team_possession_reward,
                opponent_possession_penalty=config.opponent_possession_penalty,
                successful_pass_reward=config.successful_pass_reward,
                progressive_pass_reward_scale=config.progressive_pass_reward_scale,
                carry_progress_reward_scale=config.carry_progress_reward_scale,
                attacking_third_reward=config.attacking_third_reward,
                shots_with_ball_reward=config.shots_with_ball_reward,
                attacking_risk_x_threshold=config.attacking_risk_x_threshold,
                attacking_loss_penalty_scale=config.attacking_loss_penalty_scale,
                out_of_play_loss_penalty=config.out_of_play_loss_penalty,
            ),
        )
        self.env = ReducedActionFootballEnv(env_config, render=config.render, logdir=config.logdir)
        self.model = StructuredSimple115ActorCritic(
            obs_dim=self.env.obs_dim,
            action_dim=self.env.action_dim,
            config=SaltyFishModelConfig(head_dim=config.head_dim, trunk_dim=config.trunk_dim),
        ).to(self.device)
        if config.init_checkpoint:
            checkpoint = torch.load(config.init_checkpoint, map_location="cpu", weights_only=False)
            loaded, skipped = _load_compatible_state_dict(self.model, checkpoint["model_state_dict"])
            print(
                f"Loaded {loaded} compatible parameter tensors from {config.init_checkpoint}; "
                f"skipped {skipped} incompatible tensors."
            )
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
        completed_scores: list[tuple[int, int]] = []
        event_sums = {
            "possession_gains": 0.0,
            "possession_losses": 0.0,
            "team_possession_steps": 0.0,
            "opponent_possession_steps": 0.0,
            "successful_passes": 0.0,
            "forward_ball_progress": 0.0,
            "carry_progress": 0.0,
            "attacking_third_possession_steps": 0.0,
            "shot_actions": 0.0,
            "shots_with_ball": 0.0,
            "ball_out_losses": 0.0,
            "possession_reward": 0.0,
            "pass_reward": 0.0,
            "carry_reward": 0.0,
            "territory_reward": 0.0,
            "shot_reward": 0.0,
            "out_penalty": 0.0,
            "shaping_reward": 0.0,
        }
        steps_left_values: list[float] = []

        for step in range(steps):
            obs_buffer[step] = observation
            obs_tensor = torch.as_tensor(observation, dtype=torch.float32, device=self.device)
            with torch.no_grad():
                action, logprob, value = self.model.act(obs_tensor)
            action_np = action.cpu().numpy()
            next_observation, reward, done, info = self.env.step(action_np)

            action_buffer[step] = action_np
            logprob_buffer[step] = logprob.cpu().numpy()
            value_buffer[step] = value.cpu().numpy()
            reward_buffer[step] = np.asarray(reward, dtype=np.float32).reshape(num_players)
            done_buffer[step] = float(done)
            for key in event_sums:
                event_sums[key] += float(info.get(key, 0.0))
            steps_left = float(info.get("steps_left", float("nan")))
            if not np.isnan(steps_left):
                steps_left_values.append(steps_left)

            self.current_episode_return += float(np.mean(reward))
            self.current_episode_length += 1
            self.total_agent_steps += num_players
            observation = next_observation

            if done:
                score = self.env.get_score()
                if score is not None:
                    completed_scores.append((int(score[0]), int(score[1])))
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
            last_values=last_values.reshape(num_players),
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
        denom = float(max(1, steps * num_players))
        metrics = {
            "episodes_finished": float(len(completed_returns)),
            "mean_episode_return": float(np.mean(completed_returns)) if completed_returns else float("nan"),
            "mean_episode_length": float(np.mean(completed_lengths)) if completed_lengths else float("nan"),
            "mean_goals_for": float(np.mean([left for left, _ in completed_scores])) if completed_scores else float("nan"),
            "mean_goals_against": float(np.mean([right for _, right in completed_scores])) if completed_scores else float("nan"),
            "mean_goal_diff": (
                float(np.mean([left - right for left, right in completed_scores])) if completed_scores else float("nan")
            ),
            "success_rate": (
                float(np.mean([1.0 if left > right else 0.0 for left, right in completed_scores]))
                if completed_scores
                else float("nan")
            ),
            "score_examples": ",".join(f"{left}-{right}" for left, right in completed_scores[-3:]) if completed_scores else "n/a",
            "mean_step_reward": float(np.mean(reward_buffer)),
            "mean_steps_left": float(np.mean(steps_left_values)) if steps_left_values else float("nan"),
            "min_steps_left": float(np.min(steps_left_values)) if steps_left_values else float("nan"),
            "possession_gains": event_sums["possession_gains"],
            "possession_losses": event_sums["possession_losses"],
            "team_possession_rate": event_sums["team_possession_steps"] / denom,
            "opponent_possession_rate": event_sums["opponent_possession_steps"] / denom,
            "successful_passes": event_sums["successful_passes"],
            "forward_ball_progress": event_sums["forward_ball_progress"],
            "carry_progress": event_sums["carry_progress"],
            "attacking_third_possession_rate": event_sums["attacking_third_possession_steps"] / denom,
            "shot_actions": event_sums["shot_actions"],
            "shots_with_ball": event_sums["shots_with_ball"],
            "ball_out_losses": event_sums["ball_out_losses"],
            "mean_possession_reward_per_step": event_sums["possession_reward"] / denom,
            "mean_pass_reward_per_step": event_sums["pass_reward"] / denom,
            "mean_carry_reward_per_step": event_sums["carry_reward"] / denom,
            "mean_territory_reward_per_step": event_sums["territory_reward"] / denom,
            "mean_shot_reward_per_step": event_sums["shot_reward"] / denom,
            "mean_out_penalty_per_step": event_sums["out_penalty"] / denom,
            "mean_shaping_reward_per_step": event_sums["shaping_reward"] / denom,
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

        for _ in range(self.config.update_epochs):
            indices = np.random.permutation(batch_size)
            for start in range(0, batch_size, minibatch_size):
                mb = indices[start : start + minibatch_size]
                _, new_logprob, entropy, new_value = self.model.get_action_and_value(obs[mb], actions[mb])
                logratio = new_logprob - old_logprobs[mb]
                ratio = logratio.exp()
                with torch.no_grad():
                    approx_kl = ((ratio - 1.0) - logratio).mean()

                adv = advantages[mb]
                policy_loss = torch.max(
                    -adv * ratio,
                    -adv * torch.clamp(ratio, 1.0 - self.config.clip_coef, 1.0 + self.config.clip_coef),
                ).mean()

                value_delta = new_value - old_values[mb]
                value_clipped = old_values[mb] + torch.clamp(value_delta, -self.config.clip_coef, self.config.clip_coef)
                value_loss_unclipped = (new_value - returns[mb]) ** 2
                value_loss_clipped = (value_clipped - returns[mb]) ** 2
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
        }

    def save_checkpoint(self, name: str, update: int) -> Path:
        path = self.save_dir / name
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
                "action_map": self.env.action_map.tolist(),
            },
            path,
        )
        return path

    def train(self) -> Path:
        steps_per_update = self.config.rollout_steps * self.env.num_players
        num_updates = math.ceil(self.config.total_timesteps / steps_per_update)
        observation = self.env.reset()
        latest_checkpoint = self.save_dir / "latest.pt"
        for update in range(1, num_updates + 1):
            observation, batch, rollout_metrics = self.collect_rollout(observation)
            update_metrics = self.update(batch)
            if update % self.config.log_interval == 0:
                return_text = (
                    f"{rollout_metrics['mean_episode_return']:.3f}"
                    if math.isfinite(rollout_metrics["mean_episode_return"])
                    else "n/a"
                )
                goal_diff_text = (
                    f"{rollout_metrics['mean_goal_diff']:.3f}"
                    if math.isfinite(rollout_metrics["mean_goal_diff"])
                    else "n/a"
                )
                success_rate_text = (
                    f"{rollout_metrics['success_rate']:.3f}"
                    if math.isfinite(rollout_metrics["success_rate"])
                    else "n/a"
                )
                mean_steps_left_text = (
                    f"{rollout_metrics['mean_steps_left']:.1f}"
                    if math.isfinite(rollout_metrics["mean_steps_left"])
                    else "n/a"
                )
                min_steps_left_text = (
                    f"{rollout_metrics['min_steps_left']:.0f}"
                    if math.isfinite(rollout_metrics["min_steps_left"])
                    else "n/a"
                )
                print(
                    f"[update {update}/{num_updates}] "
                    f"agent_steps={self.total_agent_steps} "
                    f"policy_loss={update_metrics['policy_loss']:.4f} "
                    f"value_loss={update_metrics['value_loss']:.4f} "
                    f"entropy={update_metrics['entropy']:.4f} "
                    f"approx_kl={update_metrics['approx_kl']:.5f} "
                    f"episodes_finished={int(rollout_metrics['episodes_finished'])} "
                    f"mean_step_reward={rollout_metrics['mean_step_reward']:.4f} "
                    f"episode_return={return_text} "
                    f"goal_diff={goal_diff_text} "
                    f"goals_for={rollout_metrics['mean_goals_for']:.2f} "
                    f"goals_against={rollout_metrics['mean_goals_against']:.2f} "
                    f"success_rate={success_rate_text} "
                    f"score_examples={rollout_metrics['score_examples']} "
                    f"mean_steps_left={mean_steps_left_text} "
                    f"min_steps_left={min_steps_left_text} "
                    f"passes={rollout_metrics['successful_passes']:.0f} "
                    f"ball_prog={rollout_metrics['forward_ball_progress']:.3f} "
                    f"carry_prog={rollout_metrics['carry_progress']:.3f} "
                    f"attack_third={rollout_metrics['attacking_third_possession_rate']:.3f} "
                    f"shots={rollout_metrics['shot_actions']:.0f} "
                    f"shots_on_ball={rollout_metrics['shots_with_ball']:.0f} "
                    f"shape_step={rollout_metrics['mean_shaping_reward_per_step']:.4f}"
                )
            if update % self.config.save_interval == 0:
                latest_checkpoint = self.save_checkpoint(f"update_{update}.pt", update)
        latest_checkpoint = self.save_checkpoint("latest.pt", num_updates)
        self.env.close()
        print(f"Training finished. Latest checkpoint: {latest_checkpoint}")
        return latest_checkpoint


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train SaltyFish-style 5v5 baseline.")
    parser.add_argument("--env-name", default="5_vs_5")
    parser.add_argument("--representation", default="simple115v2")
    parser.add_argument("--rewards", default="scoring")
    parser.add_argument("--num-controlled-players", type=int, default=4)
    parser.add_argument("--total-timesteps", type=int, default=204800)
    parser.add_argument("--rollout-steps", type=int, default=1024)
    parser.add_argument("--update-epochs", type=int, default=4)
    parser.add_argument("--num-minibatches", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--gamma", type=float, default=0.993)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-coef", type=float, default=0.2)
    parser.add_argument("--ent-coef", type=float, default=0.01)
    parser.add_argument("--vf-coef", type=float, default=0.5)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--head-dim", type=int, default=64)
    parser.add_argument("--trunk-dim", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--save-interval", type=int, default=10)
    parser.add_argument("--log-interval", type=int, default=1)
    parser.add_argument("--save-dir", default="X_Jiang/checkpoints/saltyfish_five_v_five")
    parser.add_argument("--logdir", default="X_Jiang/logs/saltyfish_five_v_five")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--init-checkpoint")
    parser.add_argument("--no-engineered-features", dest="use_engineered_features", action="store_false")
    parser.add_argument("--possession-gain-reward", type=float, default=0.2)
    parser.add_argument("--possession-loss-penalty", type=float, default=0.2)
    parser.add_argument("--team-possession-reward", type=float, default=0.001)
    parser.add_argument("--opponent-possession-penalty", type=float, default=0.001)
    parser.add_argument("--successful-pass-reward", type=float, default=0.02)
    parser.add_argument("--progressive-pass-reward-scale", type=float, default=0.05)
    parser.add_argument("--carry-progress-reward-scale", type=float, default=0.04)
    parser.add_argument("--attacking-third-reward", type=float, default=0.002)
    parser.add_argument("--shots-with-ball-reward", type=float, default=0.01)
    parser.add_argument("--attacking-risk-x-threshold", type=float, default=0.4)
    parser.add_argument("--attacking-loss-penalty-scale", type=float, default=0.5)
    parser.add_argument("--out-of-play-loss-penalty", type=float, default=0.05)
    parser.set_defaults(use_engineered_features=True)
    return parser


def main(argv: Sequence[str] | None = None) -> Path:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    trainer = SaltyFishPPOTrainer(SaltyFishPPOConfig(**vars(args)))
    return trainer.train()
