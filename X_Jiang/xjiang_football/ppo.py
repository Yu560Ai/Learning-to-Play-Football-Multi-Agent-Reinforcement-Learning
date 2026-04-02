from __future__ import annotations

import math
import random
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from presets import TrainConfig, build_config
from xjiang_football.envs import FootballEnvWrapper
from xjiang_football.model import ActorCritic
from xjiang_football.rewards import RewardShapingConfig
from xjiang_football.utils import TACTICAL_MODE_NAMES, TACTICAL_ACTION_NAMES, tactical_action_name

ATTACK_REWARD_FIELDS = (
    "attack_space_reward",
    "progressive_pass_choice_reward",
    "progressive_pass_result_reward_scale",
    "carry_progress_reward_scale",
    "zone_entry_progress_reward",
    "shot_choice_reward",
    "shot_execution_reward",
    "support_forward_lane_reward",
)

COVER_REWARD_FIELDS = (
    "recover_shape_reward",
    "second_player_support_reward",
    "hold_shape_reward",
)

ANTI_LOOP_REWARD_FIELDS = (
    "on_ball_stall_penalty",
    "on_ball_backward_drift_penalty",
    "on_ball_lateral_zigzag_penalty",
    "missed_shot_window_penalty",
    "non_emergency_clear_penalty",
    "unnecessary_goalkeeper_reset_penalty",
    "safe_reset_overuse_penalty",
)


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

        self.env = FootballEnvWrapper(
            env_name=config.env_name,
            representation=config.representation,
            rewards=config.rewards,
            render=config.render,
            logdir=config.logdir,
            num_controlled_players=config.num_controlled_players,
            channel_dimensions=config.channel_dimensions,
            reward_shaping=RewardShapingConfig(
                enabled=config.use_reward_shaping,
                closest_player_to_ball_reward=config.closest_player_to_ball_reward,
                first_defender_pressure_reward=config.first_defender_pressure_reward,
                second_player_support_reward=config.second_player_support_reward,
                recover_shape_reward=config.recover_shape_reward,
                hold_shape_reward=config.hold_shape_reward,
                ball_watch_penalty=config.ball_watch_penalty,
                idle_wander_penalty=config.idle_wander_penalty,
                goalkeeper_home_reward=config.goalkeeper_home_reward,
                goalkeeper_wander_penalty=config.goalkeeper_wander_penalty,
                possession_support_reward=config.possession_support_reward,
                attack_space_reward=config.attack_space_reward,
                progressive_pass_choice_reward=config.progressive_pass_choice_reward,
                progressive_pass_result_reward_scale=config.progressive_pass_result_reward_scale,
                carry_progress_reward_scale=config.carry_progress_reward_scale,
                zone_entry_progress_reward=config.zone_entry_progress_reward,
                safe_reset_pass_reward=config.safe_reset_pass_reward,
                backward_gk_pass_penalty=config.backward_gk_pass_penalty,
                unnecessary_goalkeeper_reset_penalty=config.unnecessary_goalkeeper_reset_penalty,
                non_emergency_clear_penalty=config.non_emergency_clear_penalty,
                shot_choice_reward=config.shot_choice_reward,
                missed_shot_window_penalty=config.missed_shot_window_penalty,
                shot_execution_reward=config.shot_execution_reward,
                on_ball_stall_penalty=config.on_ball_stall_penalty,
                on_ball_backward_drift_penalty=config.on_ball_backward_drift_penalty,
                on_ball_lateral_zigzag_penalty=config.on_ball_lateral_zigzag_penalty,
                support_spacing_reward=config.support_spacing_reward,
                support_spacing_penalty=config.support_spacing_penalty,
                support_forward_lane_reward=config.support_forward_lane_reward,
                support_static_penalty=config.support_static_penalty,
                safe_reset_overuse_penalty=config.safe_reset_overuse_penalty,
                support_behind_ball_penalty=config.support_behind_ball_penalty,
            ),
        )
        print(
            f"[startup] env ready obs_dim={self.env.obs_dim} action_dim={self.env.action_dim} "
            f"num_players={self.env.num_players} roles={self.env.get_role_names()}",
            flush=True,
        )

        self.model = ActorCritic(
            obs_dim=self.env.obs_dim,
            action_dim=self.env.action_dim,
            hidden_sizes=config.hidden_sizes,
            obs_shape=self.env.obs_shape,
            model_type=config.model_type,
            feature_dim=config.feature_dim,
            use_specialized_policy_heads=config.use_specialized_policy_heads,
            use_specialized_value_heads=config.use_specialized_value_heads,
        ).to(self.device)
        print(
            f"[startup] model ready type={config.model_type} hidden_sizes={tuple(config.hidden_sizes)}",
            flush=True,
        )

        self.base_reward_shaping = asdict(self.env.reward_shaping)
        self.adaptive_reward_scales = {"attack": 1.0, "cover": 1.0, "anti_loop": 1.0}
        self.adaptive_reward_history: list[dict[str, Any]] = []
        if config.init_checkpoint is not None:
            self.load_initial_checkpoint(config.init_checkpoint)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.learning_rate, eps=1e-5)
        self.save_dir = Path(config.save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        Path(config.logdir).mkdir(parents=True, exist_ok=True)

        self.total_agent_steps = 0
        self.current_episode_return = 0.0
        self.current_episode_length = 0

    def load_initial_checkpoint(self, checkpoint_path: str) -> None:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        self.model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        reward_state = checkpoint.get("reward_shaping_state")
        if isinstance(reward_state, dict):
            self.env.reward_shaping = RewardShapingConfig(**reward_state)
            self.base_reward_shaping = asdict(self.env.reward_shaping)
        adaptive_state = checkpoint.get("adaptive_reward_scales")
        if isinstance(adaptive_state, dict):
            for key in self.adaptive_reward_scales:
                if key in adaptive_state:
                    self.adaptive_reward_scales[key] = float(adaptive_state[key])
        self._apply_adaptive_reward_scales()

    def _apply_adaptive_reward_scales(self) -> None:
        reward_state = dict(self.base_reward_shaping)
        for field_name in ATTACK_REWARD_FIELDS:
            reward_state[field_name] = self.base_reward_shaping[field_name] * self.adaptive_reward_scales["attack"]
        for field_name in COVER_REWARD_FIELDS:
            reward_state[field_name] = self.base_reward_shaping[field_name] * self.adaptive_reward_scales["cover"]
        for field_name in ANTI_LOOP_REWARD_FIELDS:
            reward_state[field_name] = self.base_reward_shaping[field_name] * self.adaptive_reward_scales["anti_loop"]
        self.env.reward_shaping = RewardShapingConfig(**reward_state)

    def _mean_metric(self, history: list[dict[str, Any]], key: str, default: float = 0.0) -> float:
        values: list[float] = []
        for item in history:
            value = item.get(key, default)
            if isinstance(value, (float, int)) and not np.isnan(float(value)):
                values.append(float(value))
        return float(np.mean(values)) if values else default

    def _fraction_from_mix(self, mix: dict[str, int], name: str) -> float:
        total = sum(int(v) for v in mix.values())
        if total <= 0:
            return 0.0
        return float(mix.get(name, 0)) / float(total)

    def _move_scale_towards_one(self, value: float, rate: float) -> float:
        if value > 1.0:
            return max(1.0, value - rate * (value - 1.0))
        return min(1.0, value + rate * (1.0 - value))

    def _maybe_adapt_reward_weights(
        self,
        update: int,
        num_updates: int,
        rollout_metrics: dict[str, Any],
    ) -> dict[str, float] | None:
        if not self.config.use_adaptive_reward_weights:
            return None

        self.adaptive_reward_history.append(rollout_metrics)
        interval = max(1, int(self.config.adaptive_reward_interval))
        if update % interval != 0:
            return None

        history = self.adaptive_reward_history[-interval:]
        progression = self._mean_metric(history, "progression_estimate", default=0.0)
        goals_for = self._mean_metric(history, "mean_goals_for", default=0.0)
        mean_step_reward = self._mean_metric(history, "mean_step_reward", default=0.0)
        mean_reward_bonus = self._mean_metric(history, "mean_reward_bonus", default=0.0)
        missed_shot_windows = self._mean_metric(history, "missed_shot_windows", default=0.0)

        mode_mix_total: dict[str, int] = defaultdict(int)
        action_mix_total: dict[str, int] = defaultdict(int)
        for item in history:
            for name, count in item.get("tactical_mode_mix", {}).items():
                mode_mix_total[name] += int(count)
            for name, count in item.get("tactical_action_mix", {}).items():
                action_mix_total[name] += int(count)

        on_ball_fraction = self._fraction_from_mix(mode_mix_total, "on_ball")
        support_attack_fraction = self._fraction_from_mix(mode_mix_total, "support_attack")
        support_cover_fraction = self._fraction_from_mix(mode_mix_total, "support_cover")
        hold_role_fraction = self._fraction_from_mix(action_mix_total, "hold_role")
        shaping_dominance = 1.0 if abs(mean_step_reward - mean_reward_bonus) < 0.0025 else 0.0

        step = float(self.config.adaptive_scale_step)
        min_scale = float(self.config.adaptive_scale_min)
        max_scale = float(self.config.adaptive_scale_max)
        training_progress = float(update) / float(max(1, num_updates))

        cover_floor = max(min_scale, 1.0 - 0.25 * training_progress)
        if self.adaptive_reward_scales["cover"] > cover_floor:
            self.adaptive_reward_scales["cover"] = max(
                cover_floor,
                self.adaptive_reward_scales["cover"] - 0.35 * step,
            )

        attack_underpowered = (
            progression < self.config.adaptive_progression_target
            and goals_for < 0.20
            and (
                on_ball_fraction < self.config.adaptive_on_ball_fraction_target
                or support_cover_fraction > self.config.adaptive_support_cover_fraction_target
                or shaping_dominance > 0.0
            )
        )
        local_optimum_loop = (
            hold_role_fraction > self.config.adaptive_hold_role_fraction_target
            or missed_shot_windows >= 2.0
            or (progression < self.config.adaptive_progression_target and support_attack_fraction < 0.22)
        )
        healthy_attack = (
            progression > self.config.adaptive_progression_target + 0.10
            and on_ball_fraction > self.config.adaptive_on_ball_fraction_target
            and support_cover_fraction < self.config.adaptive_support_cover_fraction_target - 0.08
        )

        if attack_underpowered:
            self.adaptive_reward_scales["attack"] *= 1.0 + step
            self.adaptive_reward_scales["cover"] *= 1.0 - 0.60 * step

        if local_optimum_loop:
            self.adaptive_reward_scales["anti_loop"] *= 1.0 + step
            if support_cover_fraction > on_ball_fraction:
                self.adaptive_reward_scales["cover"] *= 1.0 - 0.40 * step

        if healthy_attack:
            self.adaptive_reward_scales["attack"] = self._move_scale_towards_one(
                self.adaptive_reward_scales["attack"],
                0.35 * step,
            )
            self.adaptive_reward_scales["anti_loop"] = self._move_scale_towards_one(
                self.adaptive_reward_scales["anti_loop"],
                0.25 * step,
            )
            self.adaptive_reward_scales["cover"] = min(
                1.0,
                self.adaptive_reward_scales["cover"] + 0.20 * step,
            )

        for name, value in self.adaptive_reward_scales.items():
            self.adaptive_reward_scales[name] = float(np.clip(value, min_scale, max_scale))

        self._apply_adaptive_reward_scales()
        return {
            "attack_scale": self.adaptive_reward_scales["attack"],
            "cover_scale": self.adaptive_reward_scales["cover"],
            "anti_loop_scale": self.adaptive_reward_scales["anti_loop"],
            "window_progression": progression,
            "window_on_ball_fraction": on_ball_fraction,
            "window_support_cover_fraction": support_cover_fraction,
            "window_hold_role_fraction": hold_role_fraction,
            "window_missed_shots": missed_shot_windows,
        }

    def collect_rollout(self, observation: np.ndarray) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, Any]]:
        steps = self.config.rollout_steps
        num_players = self.env.num_players
        obs_buffer = np.zeros((steps, num_players, self.env.obs_dim), dtype=np.float32)
        action_mask_buffer = np.zeros((steps, num_players, self.env.action_dim), dtype=np.float32)
        head_index_buffer = np.zeros((steps, num_players), dtype=np.int64)
        action_buffer = np.zeros((steps, num_players), dtype=np.int64)
        logprob_buffer = np.zeros((steps, num_players), dtype=np.float32)
        reward_buffer = np.zeros((steps, num_players), dtype=np.float32)
        done_buffer = np.zeros(steps, dtype=np.float32)
        value_buffer = np.zeros((steps, num_players), dtype=np.float32)

        completed_returns: list[float] = []
        completed_lengths: list[int] = []
        completed_scores: list[tuple[int, int]] = []
        completed_successes: list[float] = []
        metric_lists: dict[str, list[float]] = defaultdict(list)
        tactical_action_counts: dict[str, int] = defaultdict(int)
        tactical_mode_counts: dict[str, int] = defaultdict(int)
        per_mode_action_counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

        for step in range(steps):
            obs_buffer[step] = observation
            action_mask = self.env.get_action_mask()
            head_indices = self.env.get_policy_head_indices()
            mode_indices = self.env.get_tactical_mode_indices()
            action_mask_buffer[step] = action_mask
            head_index_buffer[step] = head_indices
            obs_tensor = torch.as_tensor(observation, dtype=torch.float32, device=self.device)
            action_mask_tensor = torch.as_tensor(action_mask, dtype=torch.float32, device=self.device)
            head_indices_tensor = torch.as_tensor(head_indices, dtype=torch.int64, device=self.device)
            with torch.no_grad():
                actions, logprobs, values = self.model.act(
                    obs_tensor,
                    action_mask=action_mask_tensor,
                    head_indices=head_indices_tensor,
                )
            action_np = actions.cpu().numpy()
            next_observation, reward, done, info = self.env.step(action_np)

            action_buffer[step] = action_np
            logprob_buffer[step] = logprobs.cpu().numpy()
            value_buffer[step] = values.cpu().numpy()
            reward_buffer[step] = reward
            done_buffer[step] = float(done)

            for action_id in action_np.tolist():
                tactical_action_counts[tactical_action_name(int(action_id))] += 1
            for slot, mode_id in enumerate(mode_indices.tolist()):
                mode_name = TACTICAL_MODE_NAMES.get(int(mode_id), str(int(mode_id)))
                tactical_mode_counts[mode_name] += 1
                action_name = tactical_action_name(int(action_np[slot]))
                per_mode_action_counts[mode_name][action_name] += 1

            for key in (
                "goalkeeper_x",
                "active_player_distance_to_ball",
                "closest_outfield_distance_to_ball",
                "second_outfield_distance_to_ball",
                "team_spread",
                "progression_estimate",
                "pass_target_space",
                "missed_shot_windows",
                "gk_reset_events",
                "free_ball_attack_phase",
                "reward_bonus",
            ):
                value = float(info.get(key, np.nan))
                if not np.isnan(value):
                    metric_lists[key].append(value)

            self.current_episode_return += float(np.mean(reward))
            self.current_episode_length += 1
            self.total_agent_steps += num_players
            observation = next_observation

            if done:
                score = self.env.get_score()
                if score is not None:
                    completed_scores.append((int(score[0]), int(score[1])))
                    completed_successes.append(1.0 if score[0] > score[1] else 0.0)
                completed_returns.append(self.current_episode_return)
                completed_lengths.append(self.current_episode_length)
                self.current_episode_return = 0.0
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
            "action_masks": action_mask_buffer.reshape(-1, self.env.action_dim),
            "head_indices": head_index_buffer.reshape(-1),
            "actions": action_buffer.reshape(-1),
            "logprobs": logprob_buffer.reshape(-1),
            "advantages": advantages.reshape(-1),
            "returns": returns.reshape(-1),
            "values": value_buffer.reshape(-1),
        }

        metrics: dict[str, Any] = {
            "episodes_finished": float(len(completed_returns)),
            "mean_episode_return": float(np.mean(completed_returns)) if completed_returns else float("nan"),
            "mean_episode_length": float(np.mean(completed_lengths)) if completed_lengths else float("nan"),
            "mean_goals_for": float(np.mean([left for left, _ in completed_scores])) if completed_scores else float("nan"),
            "mean_goals_against": float(np.mean([right for _, right in completed_scores])) if completed_scores else float("nan"),
            "success_rate": float(np.mean(completed_successes)) if completed_successes else float("nan"),
            "score_examples": ",".join(f"{l}-{r}" for l, r in completed_scores[-3:]) if completed_scores else "n/a",
            "mean_step_reward": float(np.mean(reward_buffer)),
            "mean_reward_bonus": float(np.mean(metric_lists.get("reward_bonus", [0.0]))),
            "goalkeeper_mean_x": float(np.mean(metric_lists.get("goalkeeper_x", [np.nan]))),
            "active_player_ball_distance": float(np.mean(metric_lists.get("active_player_distance_to_ball", [np.nan]))),
            "closest_outfield_ball_distance": float(np.mean(metric_lists.get("closest_outfield_distance_to_ball", [np.nan]))),
            "second_outfield_ball_distance": float(np.mean(metric_lists.get("second_outfield_distance_to_ball", [np.nan]))),
            "team_spread": float(np.mean(metric_lists.get("team_spread", [np.nan]))),
            "progression_estimate": float(np.mean(metric_lists.get("progression_estimate", [0.0]))),
            "pass_target_space": float(np.mean(metric_lists.get("pass_target_space", [np.nan]))),
            "missed_shot_windows": float(np.sum(metric_lists.get("missed_shot_windows", [0.0]))),
            "gk_reset_events": float(np.sum(metric_lists.get("gk_reset_events", [0.0]))),
            "free_ball_attack_phase_rate": float(np.mean(metric_lists.get("free_ball_attack_phase", [0.0]))),
            "mean_valid_actions": float(np.mean(np.sum(action_mask_buffer, axis=-1))),
            "tactical_mode_mix": dict(sorted(tactical_mode_counts.items())),
            "per_mode_tactical_mix": {
                mode_name: dict(sorted(action_counts.items()))
                for mode_name, action_counts in sorted(per_mode_action_counts.items())
            },
            "tactical_action_mix": dict(sorted(tactical_action_counts.items())),
        }
        return observation, batch, metrics

    def update(self, batch: dict[str, np.ndarray]) -> dict[str, float]:
        obs = torch.as_tensor(batch["obs"], dtype=torch.float32, device=self.device)
        action_masks = torch.as_tensor(batch["action_masks"], dtype=torch.float32, device=self.device)
        head_indices = torch.as_tensor(batch["head_indices"], dtype=torch.int64, device=self.device)
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
                    action_mask=action_masks[mb_idx],
                    head_indices=head_indices[mb_idx],
                )

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
                "reward_shaping_state": asdict(self.env.reward_shaping),
                "adaptive_reward_scales": dict(self.adaptive_reward_scales),
                "obs_dim": self.env.obs_dim,
                "obs_shape": self.env.obs_shape,
                "action_dim": self.env.action_dim,
                "num_players": self.env.num_players,
                "role_names": self.env.get_role_names(),
                "update": update,
                "total_agent_steps": self.total_agent_steps,
            },
            checkpoint_path,
        )
        return checkpoint_path

    def train(self) -> Path:
        steps_per_update = self.config.rollout_steps * self.env.num_players
        num_updates = math.ceil(self.config.total_timesteps / steps_per_update)
        print(f"[startup] training begins steps_per_update={steps_per_update} num_updates={num_updates}", flush=True)
        observation = self.env.reset()
        latest_checkpoint = self.save_dir / "latest.pt"

        for update in range(1, num_updates + 1):
            observation, batch, rollout_metrics = self.collect_rollout(observation)
            update_metrics = self.update(batch)
            adaptive_metrics = self._maybe_adapt_reward_weights(update, num_updates, rollout_metrics)
            if update % self.config.log_interval == 0:
                mix = ",".join(f"{k}={v}" for k, v in rollout_metrics["tactical_action_mix"].items()) or "n/a"
                mode_mix = ",".join(f"{k}={v}" for k, v in rollout_metrics["tactical_mode_mix"].items()) or "n/a"
                per_mode_mix = " | ".join(
                    f"{mode}:" + ",".join(f"{action}={count}" for action, count in actions.items())
                    for mode, actions in rollout_metrics["per_mode_tactical_mix"].items()
                ) or "n/a"
                reward_scales = (
                    "n/a"
                    if adaptive_metrics is None and not self.config.use_adaptive_reward_weights
                    else (
                        f"attack={self.adaptive_reward_scales['attack']:.2f},"
                        f"cover={self.adaptive_reward_scales['cover']:.2f},"
                        f"anti_loop={self.adaptive_reward_scales['anti_loop']:.2f}"
                    )
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
                    f"mean_reward_bonus={rollout_metrics['mean_reward_bonus']:.4f} "
                    f"episode_return={rollout_metrics['mean_episode_return']:.3f} "
                    f"goals_for={rollout_metrics['mean_goals_for']:.2f} "
                    f"goals_against={rollout_metrics['mean_goals_against']:.2f} "
                    f"goalkeeper_avg_x={rollout_metrics['goalkeeper_mean_x']:.3f} "
                    f"closest_outfield_ball_dist={rollout_metrics['closest_outfield_ball_distance']:.3f} "
                    f"second_outfield_ball_dist={rollout_metrics['second_outfield_ball_distance']:.3f} "
                    f"team_spread={rollout_metrics['team_spread']:.3f} "
                    f"progression={rollout_metrics['progression_estimate']:.3f} "
                    f"pass_target_space={rollout_metrics['pass_target_space']:.3f} "
                    f"missed_shot_windows={rollout_metrics['missed_shot_windows']:.0f} "
                    f"gk_reset_events={rollout_metrics['gk_reset_events']:.0f} "
                    f"free_ball_attack_rate={rollout_metrics['free_ball_attack_phase_rate']:.3f} "
                    f"valid_actions={rollout_metrics['mean_valid_actions']:.2f} "
                    f"reward_scales={reward_scales} "
                    f"mode_mix={mode_mix} "
                    f"per_mode_mix={per_mode_mix} "
                    f"score_examples={rollout_metrics['score_examples']} "
                    f"success_rate={rollout_metrics['success_rate']:.3f} "
                    f"tactical_mix={mix}",
                    flush=True,
                )
            if update % self.config.save_interval == 0:
                latest_checkpoint = self.save_checkpoint(f"update_{update}.pt", update)

        latest_checkpoint = self.save_checkpoint("latest.pt", num_updates)
        self.env.close()
        print(f"Training finished. Latest checkpoint: {latest_checkpoint}", flush=True)
        return latest_checkpoint


def main(
    preset_name: str = "five_v_five_tactical_debug",
    device_override: str | None = None,
    init_checkpoint_override: str | None = None,
) -> Path:
    overrides = {}
    if device_override is not None:
        overrides["device"] = device_override
    if init_checkpoint_override is not None:
        overrides["init_checkpoint"] = init_checkpoint_override
        checkpoint = torch.load(init_checkpoint_override, map_location="cpu")
        checkpoint_config = checkpoint.get("config", {})
        for key in (
            "hidden_sizes",
            "model_type",
            "feature_dim",
            "num_controlled_players",
            "use_specialized_policy_heads",
            "use_specialized_value_heads",
        ):
            if key in checkpoint_config:
                overrides[key] = checkpoint_config[key]
    config = build_config(preset_name, overrides=overrides)
    trainer = PPOTrainer(config)
    return trainer.train()
