from __future__ import annotations

import math
import random
import time
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical, kl_divergence

from presets import TrainConfig, build_config
from xjiang_football.envs import ParallelFootballEnvWrapper
from xjiang_football.features import SIMPLE115_DIM, SIMPLE115_SLICES
from xjiang_football.model import ActorCritic, ModelConfig
from xjiang_football.priors import batch_rule_based_single_player_actions
from xjiang_football.progress import format_duration
from xjiang_football.rewards import RewardShapingConfig


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
    last_advantage = np.zeros(rewards.shape[1:], dtype=np.float32)
    for step in reversed(range(rewards.shape[0])):
        next_values = last_values if step == rewards.shape[0] - 1 else values[step + 1]
        next_non_terminal = 1.0 - dones[step]
        delta = rewards[step] + gamma * next_values * next_non_terminal - values[step]
        last_advantage = delta + gamma * gae_lambda * next_non_terminal * last_advantage
        advantages[step] = last_advantage
    returns = advantages + values
    return advantages, returns


class PPOTrainer:
    def __init__(self, config: TrainConfig) -> None:
        self.config = config
        print(f"[startup] resolving device={config.device}", flush=True)
        self.device = resolve_device(config.device)
        print(f"[startup] using device={self.device}", flush=True)
        set_seed(config.seed)

        self.env = ParallelFootballEnvWrapper(
            num_envs=config.num_envs,
            env_name=config.env_name,
            representation=config.representation,
            rewards=config.rewards,
            render=config.render,
            logdir=config.logdir,
            num_controlled_players=config.num_controlled_players,
            channel_dimensions=config.channel_dimensions,
            reward_shaping=RewardShapingConfig(
                enabled=config.use_reward_shaping,
                possession_gain_reward=config.possession_gain_reward,
                possession_loss_penalty=config.possession_loss_penalty,
                team_possession_reward=config.team_possession_reward,
                opponent_possession_penalty=config.opponent_possession_penalty,
                successful_pass_reward=config.successful_pass_reward,
                progressive_pass_reward_scale=config.progressive_pass_reward_scale,
                carry_progress_reward_scale=config.carry_progress_reward_scale,
                attacking_third_reward=config.attacking_third_reward,
                shot_with_ball_reward=config.shot_with_ball_reward,
                checkpoint_reward_scale=config.checkpoint_reward_scale,
                attacking_risk_x_threshold=config.attacking_risk_x_threshold,
                shot_reward_x_threshold=config.shot_reward_x_threshold,
                danger_zone_x_threshold=config.danger_zone_x_threshold,
                danger_zone_entry_reward=config.danger_zone_entry_reward,
                terminal_zone_x_threshold=config.terminal_zone_x_threshold,
                terminal_zone_reward=config.terminal_zone_reward,
                finish_quality_threshold=config.finish_quality_threshold,
                finish_quality_progress_reward_scale=config.finish_quality_progress_reward_scale,
                duel_reward_scale=config.duel_reward_scale,
                low_quality_shot_penalty_scale=config.low_quality_shot_penalty_scale,
                backtracking_penalty_scale=config.backtracking_penalty_scale,
                danger_zone_stall_penalty=config.danger_zone_stall_penalty,
                bad_shot_penalty=config.bad_shot_penalty,
                attacking_loss_penalty_scale=config.attacking_loss_penalty_scale,
                out_of_play_loss_penalty=config.out_of_play_loss_penalty,
            ),
            use_engineered_features=config.use_engineered_features,
            collect_feature_metrics=config.collect_feature_metrics,
            action_set=config.action_set,
            force_shoot_in_zone=config.force_shoot_in_zone,
            force_shoot_x_threshold=config.force_shoot_x_threshold,
            force_shoot_y_threshold=config.force_shoot_y_threshold,
        )
        print(
            f"[startup] env ready obs_dim={self.env.obs_dim} action_dim={self.env.action_dim} "
            f"num_players={self.env.num_players} num_envs={self.env.num_envs}",
            flush=True,
        )

        self.model = ActorCritic(
            obs_dim=self.env.obs_dim,
            action_dim=self.env.action_dim,
            num_players=self.env.num_players,
            config=ModelConfig(
                head_dim=config.head_dim,
                trunk_dim=config.trunk_dim,
                critic_dim=config.critic_dim,
            ),
        ).to(self.device)
        print(
            f"[startup] model ready trunk_dim={config.trunk_dim} simplified_value_head=true",
            flush=True,
        )
        self.prior_model: ActorCritic | None = None
        if config.init_checkpoint is not None:
            self.load_initial_checkpoint(config.init_checkpoint)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.learning_rate, eps=1e-5)
        self.save_dir = Path(config.save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        Path(config.logdir).mkdir(parents=True, exist_ok=True)

        self.total_agent_steps = 0
        self.current_episode_return = 0.0
        self.current_episode_length = 0
        self.best_checkpoint_score: tuple[float, float, float] | None = None

    def load_initial_checkpoint(self, checkpoint_path: str) -> None:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        checkpoint_state = checkpoint["model_state_dict"]
        model_state = self.model.state_dict()
        compatible_state = {
            key: value
            for key, value in checkpoint_state.items()
            if key in model_state and model_state[key].shape == value.shape
        }
        self.model.load_state_dict(compatible_state, strict=False)
        if self.config.prior_kl_coef > 0.0 and self.config.prior_kl_decay_updates > 0:
            self.prior_model = ActorCritic(
                obs_dim=self.env.obs_dim,
                action_dim=self.env.action_dim,
                num_players=self.env.num_players,
                config=ModelConfig(
                    head_dim=self.config.head_dim,
                    trunk_dim=self.config.trunk_dim,
                    critic_dim=self.config.critic_dim,
                ),
            ).to(self.device)
            prior_state = self.prior_model.state_dict()
            compatible_prior_state = {
                key: value
                for key, value in checkpoint_state.items()
                if key in prior_state and prior_state[key].shape == value.shape
            }
            self.prior_model.load_state_dict(compatible_prior_state, strict=False)
            self.prior_model.eval()
            for parameter in self.prior_model.parameters():
                parameter.requires_grad_(False)

    def collect_rollout(
        self,
        observation: np.ndarray,
        *,
        update: int | None = None,
        num_updates: int | None = None,
    ) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, Any]]:
        steps = self.config.rollout_steps
        num_envs = self.env.num_envs
        num_players = self.env.num_players
        obs_buffer = np.zeros((steps, num_envs, num_players, self.env.obs_dim), dtype=np.float32)
        action_buffer = np.zeros((steps, num_envs, num_players), dtype=np.int64)
        logprob_buffer = np.zeros((steps, num_envs, num_players), dtype=np.float32)
        reward_buffer = np.zeros((steps, num_envs, num_players), dtype=np.float32)
        done_buffer = np.zeros((steps, num_envs, num_players), dtype=np.float32)
        value_buffer = np.zeros((steps, num_envs, num_players), dtype=np.float32)

        completed_returns: list[float] = []
        completed_scores: list[tuple[int, int]] = []
        metric_sums: dict[str, float] = defaultdict(float)
        episode_returns = np.zeros(num_envs, dtype=np.float32)
        episode_lengths = np.zeros(num_envs, dtype=np.int32)
        for step in range(steps):
            obs_buffer[step] = observation

            flat_observation = observation.reshape(num_envs * num_players, self.env.obs_dim)
            obs_tensor = torch.as_tensor(flat_observation, dtype=torch.float32, device=self.device)
            with torch.no_grad():
                actions, logprobs, values = self.model.act(obs_tensor)
            action_np = actions.cpu().numpy().reshape(num_envs, num_players)
            next_observation, reward, done, infos = self.env.step(action_np)

            action_buffer[step] = action_np
            logprob_buffer[step] = logprobs.cpu().numpy().reshape(num_envs, num_players)
            value_buffer[step] = values.cpu().numpy().reshape(num_envs, num_players)
            reward_buffer[step] = np.asarray(reward, dtype=np.float32).reshape(num_envs, num_players)
            done_expanded = np.repeat(np.asarray(done, dtype=np.float32)[:, None], num_players, axis=1)
            done_buffer[step] = done_expanded

            for env_index, info in enumerate(infos):
                for key, value in info.items():
                    if key.startswith("_"):
                        continue
                    if isinstance(value, (int, float)) and not np.isnan(float(value)):
                        metric_sums[key] += float(value)

                episode_returns[env_index] += float(np.mean(reward[env_index]))
                episode_lengths[env_index] += 1
                if done[env_index]:
                    completed_returns.append(float(episode_returns[env_index]))
                    score = info.get("_episode_score")
                    if score is not None:
                        completed_scores.append((int(score[0]), int(score[1])))
                    episode_returns[env_index] = 0.0
                    episode_lengths[env_index] = 0

            self.total_agent_steps += num_envs * num_players
            observation = next_observation

        with torch.no_grad():
            last_values = self.model.get_value(
                torch.as_tensor(observation.reshape(num_envs * num_players, self.env.obs_dim), dtype=torch.float32, device=self.device),
            ).cpu().numpy().reshape(num_envs, num_players)

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
        if self.env.num_players == 1 and self.config.bc_coef > 0.0 and self.config.bc_decay_updates > 0:
            batch["bc_actions"] = batch_rule_based_single_player_actions(
                batch["obs"],
                shoot_x_threshold=self.config.bc_shoot_x_threshold,
                shoot_y_threshold=self.config.bc_shoot_y_threshold,
            )

        denom = float(max(1, steps * num_envs * num_players))
        metrics: dict[str, Any] = {
            "episodes_finished": float(len(completed_returns)),
            "mean_episode_return": float(np.mean(completed_returns)) if completed_returns else float("nan"),
            "mean_goals_for": float(np.mean([left for left, _ in completed_scores])) if completed_scores else float("nan"),
            "mean_goals_against": float(np.mean([right for _, right in completed_scores])) if completed_scores else float("nan"),
            "success_rate": (
                float(np.mean([1.0 if left > right else 0.0 for left, right in completed_scores]))
                if completed_scores
                else float("nan")
            ),
            "score_examples": ",".join(f"{l}-{r}" for l, r in completed_scores[-3:]) if completed_scores else "n/a",
            "mean_step_reward": float(np.mean(reward_buffer)),
            "possession_gains": metric_sums["possession_gains"],
            "possession_losses": metric_sums["possession_losses"],
            "team_possession_rate": metric_sums["team_possession_steps"] / denom,
            "opponent_possession_rate": metric_sums["opponent_possession_steps"] / denom,
            "successful_passes": metric_sums["successful_passes"],
            "forward_ball_progress": metric_sums["forward_ball_progress"],
            "checkpoint_progress": metric_sums["checkpoint_progress"],
            "carry_progress": metric_sums["carry_progress"],
            "attacking_third_possession_rate": metric_sums["attacking_third_possession_steps"] / denom,
            "danger_zone_entries": metric_sums["danger_zone_entries"],
            "terminal_zone_entries": metric_sums["terminal_zone_entries"],
            "duel_engagements": metric_sums["duel_engagements"],
            "duel_beats": metric_sums["duel_beats"],
            "shot_actions": metric_sums["shot_actions"],
            "forced_shot_overrides": metric_sums["forced_shot_overrides"],
            "mean_reward_bonus": metric_sums["reward_bonus"] / float(max(1, steps)),
            "mean_possession_reward_per_step": metric_sums["possession_reward"] / denom,
            "mean_checkpoint_reward_per_step": metric_sums["checkpoint_reward"] / denom,
            "mean_pass_reward_per_step": metric_sums["pass_reward"] / denom,
            "mean_carry_reward_per_step": metric_sums["carry_reward"] / denom,
            "mean_territory_reward_per_step": metric_sums["territory_reward"] / denom,
            "mean_danger_zone_reward_per_step": metric_sums["danger_zone_reward"] / denom,
            "mean_terminal_zone_reward_per_step": metric_sums["terminal_zone_reward"] / denom,
            "mean_shot_reward_per_step": metric_sums["shot_reward"] / denom,
            "mean_finish_quality_per_step": metric_sums["finish_quality"] / denom,
            "mean_finish_quality_progress_per_step": metric_sums["finish_quality_progress"] / denom,
            "mean_finish_quality_progress_reward_per_step": metric_sums["finish_quality_progress_reward"] / denom,
            "mean_duel_reward_per_step": metric_sums["duel_reward"] / denom,
            "mean_backtracking_penalty_per_step": metric_sums["backtracking_penalty"] / denom,
            "mean_stall_penalty_per_step": metric_sums["stall_penalty"] / denom,
            "mean_bad_shot_penalty_per_step": metric_sums["bad_shot_penalty"] / denom,
            "mean_low_quality_shot_penalty_per_step": metric_sums["low_quality_shot_penalty"] / denom,
            "mean_out_penalty_per_step": metric_sums["out_penalty"] / denom,
        }
        return observation, batch, metrics

    def update(self, batch: dict[str, np.ndarray], *, update_index: int) -> dict[str, float]:
        obs = torch.as_tensor(batch["obs"], dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(batch["actions"], dtype=torch.int64, device=self.device)
        old_logprobs = torch.as_tensor(batch["logprobs"], dtype=torch.float32, device=self.device)
        advantages = torch.as_tensor(batch["advantages"], dtype=torch.float32, device=self.device)
        returns = torch.as_tensor(batch["returns"], dtype=torch.float32, device=self.device)
        old_values = torch.as_tensor(batch["values"], dtype=torch.float32, device=self.device)
        bc_actions = None
        if "bc_actions" in batch:
            bc_actions = torch.as_tensor(batch["bc_actions"], dtype=torch.int64, device=self.device)
        prior_kl_coef = 0.0
        if self.prior_model is not None and self.config.prior_kl_decay_updates > 0:
            decay_progress = min(1.0, float(max(0, update_index - 1)) / float(self.config.prior_kl_decay_updates))
            prior_kl_coef = self.config.prior_kl_coef * (1.0 - decay_progress)
        bc_coef = 0.0
        if bc_actions is not None and self.config.bc_decay_updates > 0:
            decay_progress = min(1.0, float(max(0, update_index - 1)) / float(self.config.bc_decay_updates))
            bc_coef = self.config.bc_coef * (1.0 - decay_progress)

        if advantages.numel() > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)

        possession_mask = None
        if self.env.num_players == 1 and obs.shape[1] >= SIMPLE115_DIM:
            owner_slice = obs[:, SIMPLE115_SLICES.ball_owner]
            owner_index = torch.argmax(owner_slice, dim=-1)
            possession_mask = owner_index == 1

        batch_size = obs.shape[0]
        minibatch_size = batch_size // self.config.num_minibatches
        if minibatch_size < 1:
            raise ValueError("num_minibatches is too large for the collected rollout.")

        policy_losses: list[float] = []
        value_losses: list[float] = []
        entropies: list[float] = []
        approx_kls: list[float] = []
        clipfracs: list[float] = []
        prior_kls: list[float] = []
        bc_losses: list[float] = []

        for _ in range(self.config.update_epochs):
            indices = np.random.permutation(batch_size)
            for start in range(0, batch_size, minibatch_size):
                mb_idx = indices[start : start + minibatch_size]
                logits, actor_features = self.model.actor_forward(obs[mb_idx])
                dist = Categorical(logits=logits)
                action_batch = actions[mb_idx]
                new_logprob = dist.log_prob(action_batch)
                entropy = dist.entropy()
                new_value = self.model.value_head(actor_features).squeeze(-1)

                logratio = new_logprob - old_logprobs[mb_idx]
                ratio = logratio.exp()
                with torch.no_grad():
                    approx_kl = ((ratio - 1.0) - logratio).mean()
                    clipfracs.append(float(((ratio - 1.0).abs() > self.config.clip_coef).float().mean().item()))

                mb_advantages = advantages[mb_idx]
                policy_loss_1 = -mb_advantages * ratio
                policy_loss_2 = -mb_advantages * torch.clamp(ratio, 1.0 - self.config.clip_coef, 1.0 + self.config.clip_coef)
                policy_loss = torch.max(policy_loss_1, policy_loss_2).mean()

                value_delta = new_value - old_values[mb_idx]
                value_clipped = old_values[mb_idx] + torch.clamp(value_delta, -self.config.clip_coef, self.config.clip_coef)
                value_loss_unclipped = (new_value - returns[mb_idx]) ** 2
                value_loss_clipped = (value_clipped - returns[mb_idx]) ** 2
                value_loss = 0.5 * torch.max(value_loss_unclipped, value_loss_clipped).mean()

                entropy_loss = entropy.mean()
                prior_kl_loss = torch.zeros((), device=self.device)
                if prior_kl_coef > 0.0 and self.prior_model is not None:
                    with torch.no_grad():
                        prior_logits, _ = self.prior_model.actor_forward(obs[mb_idx])
                    prior_dist = Categorical(logits=prior_logits)
                    prior_kl_values = kl_divergence(prior_dist, dist)
                    if possession_mask is not None:
                        mb_possession_mask = possession_mask[mb_idx]
                        if torch.any(mb_possession_mask):
                            prior_kl_loss = prior_kl_values[mb_possession_mask].mean()
                    else:
                        prior_kl_loss = prior_kl_values.mean()
                bc_loss = torch.zeros((), device=self.device)
                if bc_coef > 0.0 and bc_actions is not None:
                    if possession_mask is not None:
                        mb_possession_mask = possession_mask[mb_idx]
                        if torch.any(mb_possession_mask):
                            bc_loss = nn.functional.cross_entropy(
                                logits[mb_possession_mask],
                                bc_actions[mb_idx][mb_possession_mask],
                            )
                    else:
                        bc_loss = nn.functional.cross_entropy(logits, bc_actions[mb_idx])

                loss = (
                    policy_loss
                    + self.config.vf_coef * value_loss
                    - self.config.ent_coef * entropy_loss
                    + prior_kl_coef * prior_kl_loss
                    + bc_coef * bc_loss
                )

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                self.optimizer.step()

                policy_losses.append(float(policy_loss.item()))
                value_losses.append(float(value_loss.item()))
                entropies.append(float(entropy_loss.item()))
                approx_kls.append(float(approx_kl.item()))
                prior_kls.append(float(prior_kl_loss.item()))
                bc_losses.append(float(bc_loss.item()))

        return {
            "policy_loss": float(np.mean(policy_losses)),
            "value_loss": float(np.mean(value_losses)),
            "entropy": float(np.mean(entropies)),
            "approx_kl": float(np.mean(approx_kls)),
            "clipfrac": float(np.mean(clipfracs)) if clipfracs else 0.0,
            "prior_kl": float(np.mean(prior_kls)) if prior_kls else 0.0,
            "prior_kl_coef": float(prior_kl_coef),
            "bc_loss": float(np.mean(bc_losses)) if bc_losses else 0.0,
            "bc_coef": float(bc_coef),
        }

    def save_checkpoint(self, name: str, update: int) -> Path:
        checkpoint_path = self.save_dir / name
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "config": asdict(self.config),
                "reward_shaping_state": asdict(self.env.reward_shaping),
                "obs_dim": self.env.obs_dim,
                "obs_shape": self.env.obs_shape,
                "action_dim": self.env.action_dim,
                "num_players": self.env.num_players,
                "action_names": self.env.action_names,
                "update": update,
                "total_agent_steps": self.total_agent_steps,
            },
            checkpoint_path,
        )
        return checkpoint_path

    def _rollout_score_tuple(self, rollout_metrics: dict[str, Any]) -> tuple[float, float, float]:
        goals_for = float(rollout_metrics["mean_goals_for"]) if not np.isnan(float(rollout_metrics["mean_goals_for"])) else -1.0
        success_rate = float(rollout_metrics["success_rate"]) if not np.isnan(float(rollout_metrics["success_rate"])) else -1.0
        episode_return = float(rollout_metrics["mean_episode_return"]) if not np.isnan(float(rollout_metrics["mean_episode_return"])) else -1e9
        return (goals_for, success_rate, episode_return)

    def train(self) -> Path:
        steps_per_update = self.config.rollout_steps * self.env.num_players * self.env.num_envs
        num_updates = math.ceil(self.config.total_timesteps / steps_per_update)
        print(f"[startup] training begins steps_per_update={steps_per_update} num_updates={num_updates}", flush=True)
        observation = self.env.reset()
        latest_checkpoint = self.save_dir / "latest.pt"
        train_start_time = time.monotonic()

        for update in range(1, num_updates + 1):
            observation, batch, rollout_metrics = self.collect_rollout(
                observation,
                update=update,
                num_updates=num_updates,
            )
            update_metrics = self.update(batch, update_index=update)
            if update % self.config.log_interval == 0:
                elapsed_seconds = time.monotonic() - train_start_time
                avg_update_time = elapsed_seconds / float(max(1, update))
                remaining_seconds = avg_update_time * float(max(0, num_updates - update))
                print(
                    f"[update {update}/{num_updates}] "
                    f"agent_steps={self.total_agent_steps} "
                    f"policy_loss={update_metrics['policy_loss']:.4f} "
                    f"value_loss={update_metrics['value_loss']:.4f} "
                    f"entropy={update_metrics['entropy']:.4f} "
                    f"approx_kl={update_metrics['approx_kl']:.5f} "
                    f"prior_kl={update_metrics['prior_kl']:.5f} "
                    f"prior_coef={update_metrics['prior_kl_coef']:.3f} "
                    f"bc_loss={update_metrics['bc_loss']:.4f} "
                    f"bc_coef={update_metrics['bc_coef']:.3f} "
                    f"episodes_finished={int(rollout_metrics['episodes_finished'])} "
                    f"mean_step_reward={rollout_metrics['mean_step_reward']:.4f} "
                    f"mean_reward_bonus={rollout_metrics['mean_reward_bonus']:.4f} "
                    f"episode_return={rollout_metrics['mean_episode_return']:.3f} "
                    f"goals_for={rollout_metrics['mean_goals_for']:.2f} "
                    f"goals_against={rollout_metrics['mean_goals_against']:.2f} "
                    f"success_rate={rollout_metrics['success_rate']:.3f} "
                    f"team_possession={rollout_metrics['team_possession_rate']:.3f} "
                    f"opp_possession={rollout_metrics['opponent_possession_rate']:.3f} "
                    f"passes={rollout_metrics['successful_passes']:.0f} "
                    f"ball_prog={rollout_metrics['forward_ball_progress']:.3f} "
                    f"checkpoints={rollout_metrics['checkpoint_progress']:.1f} "
                    f"danger_entries={rollout_metrics['danger_zone_entries']:.1f} "
                    f"terminal_entries={rollout_metrics['terminal_zone_entries']:.1f} "
                    f"duels={rollout_metrics['duel_engagements']:.1f} "
                    f"beat_duels={rollout_metrics['duel_beats']:.1f} "
                    f"carry_progress={rollout_metrics['carry_progress']:.3f} "
                    f"shots={rollout_metrics['shot_actions']:.0f} "
                    f"forced_shots={rollout_metrics['forced_shot_overrides']:.0f} "
                    f"attacking_third={rollout_metrics['attacking_third_possession_rate']:.3f} "
                    f"r_pos={rollout_metrics['mean_possession_reward_per_step']:.4f} "
                    f"r_chk={rollout_metrics['mean_checkpoint_reward_per_step']:.4f} "
                    f"r_pass={rollout_metrics['mean_pass_reward_per_step']:.4f} "
                    f"r_carry={rollout_metrics['mean_carry_reward_per_step']:.4f} "
                    f"r_terr={rollout_metrics['mean_territory_reward_per_step']:.4f} "
                    f"r_danger={rollout_metrics['mean_danger_zone_reward_per_step']:.4f} "
                    f"r_term={rollout_metrics['mean_terminal_zone_reward_per_step']:.4f} "
                    f"r_shot={rollout_metrics['mean_shot_reward_per_step']:.4f} "
                    f"finish_q={rollout_metrics['mean_finish_quality_per_step']:.4f} "
                    f"finish_q_prog={rollout_metrics['mean_finish_quality_progress_per_step']:.4f} "
                    f"r_finish={rollout_metrics['mean_finish_quality_progress_reward_per_step']:.4f} "
                    f"r_duel={rollout_metrics['mean_duel_reward_per_step']:.4f} "
                    f"p_back={rollout_metrics['mean_backtracking_penalty_per_step']:.4f} "
                    f"p_stall={rollout_metrics['mean_stall_penalty_per_step']:.4f} "
                    f"p_badshot={rollout_metrics['mean_bad_shot_penalty_per_step']:.4f} "
                    f"p_lowq={rollout_metrics['mean_low_quality_shot_penalty_per_step']:.4f} "
                    f"r_out={rollout_metrics['mean_out_penalty_per_step']:.4f} "
                    f"score_examples={rollout_metrics['score_examples']} "
                    f"elapsed={format_duration(elapsed_seconds)} "
                    f"remaining={format_duration(remaining_seconds)}",
                    flush=True,
                )
            rollout_score = self._rollout_score_tuple(rollout_metrics)
            if self.best_checkpoint_score is None or rollout_score > self.best_checkpoint_score:
                self.best_checkpoint_score = rollout_score
                self.save_checkpoint("best.pt", update)
                print(
                    f"[best] update={update} goals_for={rollout_score[0]:.2f} "
                    f"success_rate={rollout_score[1]:.3f} episode_return={rollout_score[2]:.3f}",
                    flush=True,
                )
            if update % self.config.save_interval == 0:
                latest_checkpoint = self.save_checkpoint(f"update_{update}.pt", update)

        latest_checkpoint = self.save_checkpoint("latest.pt", num_updates)
        self.env.close()
        print(f"Training finished. Latest checkpoint: {latest_checkpoint}", flush=True)
        return latest_checkpoint


def main(
    preset_name: str = "five_v_five_football_base",
    device_override: str | None = None,
    init_checkpoint_override: str | None = None,
    total_timesteps_override: int | None = None,
    rollout_steps_override: int | None = None,
) -> Path:
    overrides: dict[str, Any] = {}
    if device_override is not None:
        overrides["device"] = device_override
    if init_checkpoint_override is not None:
        overrides["init_checkpoint"] = init_checkpoint_override
        checkpoint = torch.load(init_checkpoint_override, map_location="cpu")
        checkpoint_config = checkpoint.get("config", {})
        # Warm-start should preserve architecture-compatible settings, but it
        # must not silently replace the new curriculum stage's environment or
        # action-space definition with the old checkpoint's config.
        for key in ("head_dim", "trunk_dim", "critic_dim", "use_engineered_features"):
            if key in checkpoint_config:
                overrides[key] = checkpoint_config[key]
    if total_timesteps_override is not None:
        overrides["total_timesteps"] = int(total_timesteps_override)
    if rollout_steps_override is not None:
        overrides["rollout_steps"] = int(rollout_steps_override)
    config = build_config(preset_name, overrides=overrides)
    trainer = PPOTrainer(config)
    return trainer.train()
