from __future__ import annotations

import json
import random
import sys
import time
from argparse import Namespace
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn


def ensure_gym_compatibility() -> None:
    try:
        import gym  # type: ignore  # noqa: F401
        return
    except ImportError:
        pass

    repo_root = Path(__file__).resolve().parents[2]
    bundled_env_root = repo_root / "football-master" / "football-env"
    for site_packages in sorted(bundled_env_root.glob("lib/python*/site-packages")):
        site_packages_str = str(site_packages)
        if site_packages.exists() and site_packages_str not in sys.path:
            sys.path.insert(0, site_packages_str)
        try:
            import gym  # type: ignore  # noqa: F401
            return
        except ImportError:
            continue

    import gymnasium as gymnasium  # type: ignore

    sys.modules.setdefault("gym", gymnasium)


def ensure_tikick_path() -> Path:
    root = Path(__file__).resolve().parents[1] / "third_party" / "tikick"
    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    return root


ensure_gym_compatibility()
ensure_tikick_path()

from tmarl.algorithms.r_mappo_distributed.mappo_module import MAPPOModule
from tmarl.envs.env_wrappers import ShareDummyVecEnv, ShareSubprocVecEnv
from tmarl.replay_buffers.normal.shared_buffer import SharedReplayBuffer
from tmarl.utils.util import get_gard_norm, huber_loss, mse_loss

from Two_V_Two.env.grf_simple_env import TwoVTwoFootballEnv


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _to_numpy(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().cpu().numpy()


def _activation_from_id(activation_id: int) -> nn.Module:
    if activation_id == 0:
        return nn.Tanh()
    if activation_id == 2:
        return nn.LeakyReLU()
    if activation_id == 3:
        return nn.ELU()
    return nn.ReLU()


class CentralizedValueNetwork(nn.Module):
    def __init__(self, args: Namespace, share_obs_space, device: torch.device):
        super().__init__()
        input_dim = int(np.prod(share_obs_space.shape))
        hidden_size = int(args.hidden_size)
        layer_count = max(1, int(args.layer_N))
        layers: list[nn.Module] = []
        last_dim = input_dim

        for _ in range(layer_count):
            linear = nn.Linear(last_dim, hidden_size)
            if bool(args.use_orthogonal):
                nn.init.orthogonal_(linear.weight)
            else:
                nn.init.xavier_uniform_(linear.weight)
            nn.init.constant_(linear.bias, 0.0)
            layers.extend([linear, _activation_from_id(int(args.activation_id))])
            last_dim = hidden_size

        value_head = nn.Linear(last_dim, 1)
        if bool(args.use_orthogonal):
            nn.init.orthogonal_(value_head.weight)
        else:
            nn.init.xavier_uniform_(value_head.weight)
        nn.init.constant_(value_head.bias, 0.0)
        layers.append(value_head)

        self.model = nn.Sequential(*layers)
        self.tpdv = dict(dtype=torch.float32, device=device)
        self.to(device)

    def forward(self, share_obs: torch.Tensor) -> torch.Tensor:
        if not isinstance(share_obs, torch.Tensor):
            share_obs = torch.as_tensor(share_obs, **self.tpdv)
        else:
            share_obs = share_obs.to(**self.tpdv)
        return self.model(share_obs)


class SharedPolicyPPOTrainer:
    def __init__(self, args: Namespace):
        self.args = args
        set_seed(int(args.seed))

        use_cuda = torch.cuda.is_available() and not bool(args.disable_cuda)
        self.device = torch.device("cuda" if use_cuda else "cpu")

        self.run_dir = Path(args.run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir = self.run_dir / "checkpoints"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_path = self.run_dir / "metrics.jsonl"
        self.config_path = self.run_dir / "config.json"
        self.config_path.write_text(json.dumps(vars(args), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        self.structure_variant = str(getattr(args, "structure_variant", "shared_ppo")).lower()
        self.use_centralized_critic = self.structure_variant == "mappo_id_cc"

        self.envs = self._make_train_envs()
        share_obs_space = self.envs.share_observation_space[0]
        obs_space = self.envs.observation_space[0]
        act_space = self.envs.action_space[0]

        self.algo_module = MAPPOModule(args, obs_space, share_obs_space, act_space, device=self.device)
        self.centralized_critic = None
        self.critic_optimizer = None
        if self.use_centralized_critic:
            self.centralized_critic = CentralizedValueNetwork(args, share_obs_space, self.device)
            self.critic_optimizer = torch.optim.Adam(
                self.centralized_critic.parameters(),
                lr=float(args.critic_lr),
                eps=float(args.opti_eps),
                weight_decay=float(args.weight_decay),
            )
        self.buffer = SharedReplayBuffer(args, args.num_agents, obs_space, share_obs_space, act_space)

        self.episode_length = int(args.episode_length)
        self.n_rollout_threads = int(args.n_rollout_threads)
        self.num_agents = int(args.num_agents)
        self.total_env_steps = 0

    def _assert_obs_shapes(self, obs: np.ndarray, share_obs: np.ndarray, available_actions: np.ndarray) -> None:
        expected_obs_shape = (self.n_rollout_threads, self.num_agents, self.envs.observation_space[0].shape[0])
        expected_share_shape = (
            self.n_rollout_threads,
            self.num_agents,
            self.envs.share_observation_space[0].shape[0],
        )
        expected_action_shape = (self.n_rollout_threads, self.num_agents, self.envs.action_space[0].n)
        if tuple(obs.shape) != expected_obs_shape:
            raise ValueError(f"Unexpected obs shape {obs.shape}, expected {expected_obs_shape}")
        if tuple(share_obs.shape) != expected_share_shape:
            raise ValueError(f"Unexpected share_obs shape {share_obs.shape}, expected {expected_share_shape}")
        if tuple(available_actions.shape) != expected_action_shape:
            raise ValueError(
                f"Unexpected available_actions shape {available_actions.shape}, expected {expected_action_shape}"
            )

    @staticmethod
    def _extract_env_info(infos: Any, env_idx: int) -> dict[str, Any]:
        env_info = infos[env_idx]
        if isinstance(env_info, np.ndarray):
            flat = env_info.reshape(-1)
            candidate = flat[0] if flat.size else {}
        elif isinstance(env_info, (list, tuple)):
            candidate = env_info[0] if env_info else {}
        else:
            candidate = env_info
        return candidate if isinstance(candidate, dict) else {}

    def _make_train_envs(self):
        def get_env_fn(rank: int):
            def init_env():
                env = TwoVTwoFootballEnv(self.args, rank, log_dir=str(self.run_dir / "replays"))
                env.seed(int(self.args.seed) + rank * 1000)
                return env

            return init_env

        if int(self.args.n_rollout_threads) == 1:
            return ShareDummyVecEnv([get_env_fn(0)])
        return ShareSubprocVecEnv([get_env_fn(rank) for rank in range(int(self.args.n_rollout_threads))])

    def _warmup(self):
        obs, share_obs, available_actions = self.envs.reset()
        self._assert_obs_shapes(obs, share_obs, available_actions)
        self.buffer.init_buffer(share_obs, obs)
        self.buffer.available_actions[0] = available_actions.copy()
        self.buffer.masks[0] = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        self.buffer.active_masks[0] = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        return obs, share_obs, available_actions

    def _value_inputs(self, obs: np.ndarray, share_obs: np.ndarray) -> np.ndarray:
        return share_obs if self.use_centralized_critic else obs

    def _predict_values(self, obs: np.ndarray, share_obs: np.ndarray, rnn_states: np.ndarray, masks: np.ndarray) -> np.ndarray:
        value_inputs = self._value_inputs(obs, share_obs)
        flat_inputs = value_inputs.reshape(-1, value_inputs.shape[-1])
        if self.use_centralized_critic:
            assert self.centralized_critic is not None
            with torch.no_grad():
                values = self.centralized_critic(flat_inputs)
        else:
            with torch.no_grad():
                values = self.algo_module.actor.get_policy_values(
                    flat_inputs,
                    rnn_states.reshape(-1, *rnn_states.shape[2:]),
                    masks.reshape(-1, 1),
                )
        return _to_numpy(values).reshape(self.n_rollout_threads, self.num_agents, 1)

    def _collect_step(self, obs, share_obs, available_actions):
        rnn_states = self.buffer.rnn_states[self.buffer.step]
        masks = self.buffer.masks[self.buffer.step]

        flat_obs = obs.reshape(-1, obs.shape[-1])
        flat_rnn_states = rnn_states.reshape(-1, *rnn_states.shape[2:])
        flat_masks = masks.reshape(-1, 1)
        flat_available_actions = available_actions.reshape(-1, available_actions.shape[-1])

        with torch.no_grad():
            actions, action_log_probs, next_rnn_states = self.algo_module.actor(
                flat_obs,
                flat_rnn_states,
                flat_masks,
                flat_available_actions,
                deterministic=False,
            )

        actions_np = _to_numpy(actions).reshape(self.n_rollout_threads, self.num_agents, -1)
        action_log_probs_np = _to_numpy(action_log_probs).reshape(self.n_rollout_threads, self.num_agents, -1)
        values_np = self._predict_values(obs, share_obs, rnn_states, masks)
        next_rnn_states_np = _to_numpy(next_rnn_states).reshape(self.n_rollout_threads, self.num_agents, *rnn_states.shape[2:])

        env_actions = actions_np.squeeze(-1).astype(np.int64)
        next_obs, next_share_obs, rewards, dones, infos, next_available_actions = self.envs.step(env_actions)
        self._assert_obs_shapes(next_obs, next_share_obs, next_available_actions)

        done_array = np.asarray(dones, dtype=np.bool_)
        done_mask = done_array[..., None]
        masks = (~done_array).astype(np.float32)[..., None]
        next_rnn_states_np[done_mask.repeat(next_rnn_states_np.shape[-2] * next_rnn_states_np.shape[-1], axis=-1).reshape(next_rnn_states_np.shape)] = 0.0

        self.buffer.insert(
            next_share_obs,
            next_obs,
            next_rnn_states_np,
            np.zeros_like(next_rnn_states_np, dtype=np.float32),
            actions_np.astype(np.float32),
            action_log_probs_np.astype(np.float32),
            values_np.astype(np.float32),
            rewards.astype(np.float32),
            masks.astype(np.float32),
            active_masks=masks.astype(np.float32),
            available_actions=next_available_actions.astype(np.float32),
        )
        return next_obs, next_share_obs, next_available_actions, rewards, done_array, infos

    def _compute_returns(self):
        last_obs = self.buffer.obs[-1]
        last_share_obs = self.buffer.share_obs[-1]
        last_rnn_states = self.buffer.rnn_states[-1]
        last_masks = self.buffer.masks[-1]
        next_values_np = self._predict_values(last_obs, last_share_obs, last_rnn_states, last_masks)
        self.buffer.compute_returns(next_values_np, value_normalizer=None)

    def _value_loss(self, values, value_preds_batch, return_batch, active_masks_batch):
        value_pred_clipped = value_preds_batch + (values - value_preds_batch).clamp(
            -self.args.clip_param,
            self.args.clip_param,
        )
        error_clipped = return_batch - value_pred_clipped
        error_original = return_batch - values
        if self.args.use_huber_loss:
            value_loss_clipped = huber_loss(error_clipped, self.args.huber_delta)
            value_loss_original = huber_loss(error_original, self.args.huber_delta)
        else:
            value_loss_clipped = mse_loss(error_clipped)
            value_loss_original = mse_loss(error_original)

        if self.args.use_clipped_value_loss:
            value_loss = torch.max(value_loss_original, value_loss_clipped)
        else:
            value_loss = value_loss_original

        if self.args.use_value_active_masks:
            return (value_loss * active_masks_batch).sum() / active_masks_batch.sum().clamp(min=1.0)
        return value_loss.mean()

    def _ppo_update(self) -> dict[str, float]:
        advantages = self.buffer.returns[:-1] - self.buffer.value_preds[:-1]
        adv_mean = advantages.mean()
        adv_std = advantages.std()
        advantages = (advantages - adv_mean) / (adv_std + 1e-5)

        policy_losses: list[float] = []
        value_losses: list[float] = []
        entropies: list[float] = []
        grad_norms: list[float] = []
        approx_kls: list[float] = []

        for _ in range(int(self.args.ppo_epoch)):
            data_generator = self.buffer.feed_forward_generator(
                advantages,
                num_mini_batch=int(self.args.num_mini_batch),
            )
            for sample in data_generator:
                (
                    share_obs_batch,
                    obs_batch,
                    rnn_states_batch,
                    _rnn_states_critic_batch,
                    actions_batch,
                    value_preds_batch,
                    return_batch,
                    masks_batch,
                    active_masks_batch,
                    old_action_log_probs_batch,
                    adv_targ,
                    available_actions_batch,
                ) = sample

                obs_batch_t = torch.as_tensor(obs_batch, dtype=torch.float32, device=self.device)
                share_obs_batch_t = torch.as_tensor(share_obs_batch, dtype=torch.float32, device=self.device)
                rnn_states_batch_t = torch.as_tensor(rnn_states_batch, dtype=torch.float32, device=self.device)
                actions_batch_t = torch.as_tensor(actions_batch, dtype=torch.float32, device=self.device)
                value_preds_batch_t = torch.as_tensor(value_preds_batch, dtype=torch.float32, device=self.device)
                return_batch_t = torch.as_tensor(return_batch, dtype=torch.float32, device=self.device)
                masks_batch_t = torch.as_tensor(masks_batch, dtype=torch.float32, device=self.device)
                active_masks_batch_t = torch.as_tensor(active_masks_batch, dtype=torch.float32, device=self.device)
                old_action_log_probs_batch_t = torch.as_tensor(
                    old_action_log_probs_batch,
                    dtype=torch.float32,
                    device=self.device,
                )
                adv_targ_t = torch.as_tensor(adv_targ, dtype=torch.float32, device=self.device)
                available_actions_batch_t = None
                if available_actions_batch is not None:
                    available_actions_batch_t = torch.as_tensor(
                        available_actions_batch,
                        dtype=torch.float32,
                        device=self.device,
                    )

                action_log_probs, dist_entropy, actor_values = self.algo_module.actor.evaluate_actions(
                    obs_batch_t,
                    rnn_states_batch_t,
                    actions_batch_t,
                    masks_batch_t,
                    available_actions_batch_t,
                    active_masks_batch_t,
                )
                if self.use_centralized_critic:
                    assert self.centralized_critic is not None
                    values = self.centralized_critic(share_obs_batch_t)
                else:
                    values = actor_values

                ratio = torch.exp(action_log_probs - old_action_log_probs_batch_t)
                surr1 = ratio * adv_targ_t
                surr2 = torch.clamp(
                    ratio,
                    1.0 - self.args.clip_param,
                    1.0 + self.args.clip_param,
                ) * adv_targ_t
                policy_loss_raw = -torch.min(surr1, surr2)
                if self.args.use_policy_active_masks:
                    policy_loss = (policy_loss_raw * active_masks_batch_t).sum() / active_masks_batch_t.sum().clamp(min=1.0)
                else:
                    policy_loss = policy_loss_raw.mean()

                value_loss = self._value_loss(values, value_preds_batch_t, return_batch_t, active_masks_batch_t)
                if self.use_centralized_critic:
                    actor_loss = policy_loss - self.args.entropy_coef * dist_entropy
                    critic_loss = self.args.value_loss_coef * value_loss

                    self.algo_module.actor_optimizer.zero_grad()
                    actor_loss.backward()
                    if self.args.use_max_grad_norm:
                        actor_grad_norm = torch.nn.utils.clip_grad_norm_(
                            self.algo_module.actor.parameters(),
                            self.args.max_grad_norm,
                        )
                        grad_norm_value = float(
                            actor_grad_norm.item() if isinstance(actor_grad_norm, torch.Tensor) else actor_grad_norm
                        )
                    else:
                        grad_norm_value = float(get_gard_norm(self.algo_module.actor.parameters()))
                    self.algo_module.actor_optimizer.step()

                    assert self.critic_optimizer is not None
                    self.critic_optimizer.zero_grad()
                    critic_loss.backward()
                    if self.args.use_max_grad_norm:
                        torch.nn.utils.clip_grad_norm_(self.centralized_critic.parameters(), self.args.max_grad_norm)
                    self.critic_optimizer.step()
                else:
                    loss = policy_loss + self.args.value_loss_coef * value_loss - self.args.entropy_coef * dist_entropy

                    self.algo_module.actor_optimizer.zero_grad()
                    loss.backward()
                    if self.args.use_max_grad_norm:
                        grad_norm = torch.nn.utils.clip_grad_norm_(
                            self.algo_module.actor.parameters(),
                            self.args.max_grad_norm,
                        )
                        grad_norm_value = float(grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm)
                    else:
                        grad_norm_value = float(get_gard_norm(self.algo_module.actor.parameters()))
                    self.algo_module.actor_optimizer.step()

                with torch.no_grad():
                    log_ratio = action_log_probs - old_action_log_probs_batch_t
                    approx_kl = ((ratio - 1.0) - log_ratio).mean()

                policy_losses.append(float(policy_loss.item()))
                value_losses.append(float(value_loss.item()))
                entropies.append(float(dist_entropy.item()))
                grad_norms.append(grad_norm_value)
                approx_kls.append(float(approx_kl.item()))

        return {
            "policy_loss": float(np.mean(policy_losses)) if policy_losses else 0.0,
            "value_loss": float(np.mean(value_losses)) if value_losses else 0.0,
            "entropy": float(np.mean(entropies)) if entropies else 0.0,
            "grad_norm": float(np.mean(grad_norms)) if grad_norms else 0.0,
            "approx_kl": float(np.mean(approx_kls)) if approx_kls else 0.0,
        }

    def _save_checkpoint(self, update: int) -> Path:
        checkpoint_path = self.checkpoint_dir / f"update_{update:06d}.pt"
        payload = {
            "actor_state_dict": self.algo_module.actor.state_dict(),
            "optimizer_state_dict": self.algo_module.actor_optimizer.state_dict(),
            "critic_state_dict": self.centralized_critic.state_dict() if self.centralized_critic is not None else None,
            "critic_optimizer_state_dict": self.critic_optimizer.state_dict() if self.critic_optimizer is not None else None,
            "args": vars(self.args),
            "update": update,
            "total_env_steps": self.total_env_steps,
        }
        torch.save(payload, checkpoint_path)
        latest_path = self.checkpoint_dir / "latest.pt"
        torch.save(payload, latest_path)
        return checkpoint_path

    def train(self) -> Path:
        obs, share_obs, available_actions = self._warmup()

        env_steps_per_update = self.episode_length * self.n_rollout_threads
        num_updates = max(1, int(np.ceil(self.args.num_env_steps / env_steps_per_update)))
        latest_checkpoint = self.checkpoint_dir / "latest.pt"
        current_returns = np.zeros(self.n_rollout_threads, dtype=np.float32)
        current_lengths = np.zeros(self.n_rollout_threads, dtype=np.int32)
        completed_returns: list[float] = []
        completed_lengths: list[int] = []
        completed_successes: list[float] = []
        completed_goal_counts: list[float] = []
        completed_pass_counts: list[float] = []
        completed_assist_counts: list[float] = []
        completed_pass_to_shot_counts: list[float] = []
        completed_possession_means: list[float] = []

        train_start = time.perf_counter()
        for update in range(1, num_updates + 1):
            rollout_start = time.perf_counter()
            for _ in range(self.episode_length):
                obs, share_obs, available_actions, rewards, dones, infos = self._collect_step(
                    obs,
                    share_obs,
                    available_actions,
                )
                mean_rewards = rewards.mean(axis=1).reshape(-1)
                current_returns += mean_rewards
                current_lengths += 1
                done_envs = dones.all(axis=1)
                for env_idx, done_flag in enumerate(done_envs):
                    if not done_flag:
                        continue
                    env_info = self._extract_env_info(infos, env_idx)
                    episode_metrics = env_info.get("episode_metrics", {})
                    completed_returns.append(float(current_returns[env_idx]))
                    completed_lengths.append(int(current_lengths[env_idx]))
                    completed_successes.append(float(current_returns[env_idx] > 0.0))
                    completed_goal_counts.append(float(episode_metrics.get("goal_count", 0.0)))
                    completed_pass_counts.append(float(episode_metrics.get("pass_count", 0.0)))
                    completed_assist_counts.append(float(episode_metrics.get("assist_count", 0.0)))
                    completed_pass_to_shot_counts.append(float(episode_metrics.get("pass_to_shot_count", 0.0)))
                    completed_possession_means.append(
                        float(episode_metrics.get("mean_same_owner_possession_length", 0.0))
                    )
                    current_returns[env_idx] = 0.0
                    current_lengths[env_idx] = 0

            rollout_time = max(time.perf_counter() - rollout_start, 1e-8)
            self.total_env_steps += env_steps_per_update
            self._compute_returns()

            update_start = time.perf_counter()
            update_metrics = self._ppo_update()
            update_time = max(time.perf_counter() - update_start, 1e-8)
            self.buffer.after_update()

            mean_return = float(np.mean(completed_returns[-100:])) if completed_returns else float("nan")
            mean_length = float(np.mean(completed_lengths[-100:])) if completed_lengths else float("nan")
            success_rate = float(np.mean(completed_successes[-100:])) if completed_successes else float("nan")
            mean_goals = float(np.mean(completed_goal_counts[-100:])) if completed_goal_counts else float("nan")
            mean_passes = float(np.mean(completed_pass_counts[-100:])) if completed_pass_counts else float("nan")
            mean_assists = float(np.mean(completed_assist_counts[-100:])) if completed_assist_counts else float("nan")
            mean_pass_to_shot = (
                float(np.mean(completed_pass_to_shot_counts[-100:])) if completed_pass_to_shot_counts else float("nan")
            )
            mean_possession = (
                float(np.mean(completed_possession_means[-100:])) if completed_possession_means else float("nan")
            )
            total_elapsed = max(time.perf_counter() - train_start, 1e-8)

            metrics = {
                "update": update,
                "env_steps": self.total_env_steps,
                "episodes_finished": len(completed_returns),
                "reward_variant": self.args.reward_variant,
                "structure_variant": self.structure_variant,
                "mean_episode_return": mean_return,
                "mean_episode_length": mean_length,
                "success_rate": success_rate,
                "mean_goal_count": mean_goals,
                "mean_pass_count": mean_passes,
                "mean_assist_count": mean_assists,
                "mean_pass_to_shot_count": mean_pass_to_shot,
                "mean_same_owner_possession_length": mean_possession,
                "rollout_fps": env_steps_per_update / rollout_time,
                "update_fps": env_steps_per_update / update_time,
                "overall_fps": self.total_env_steps / total_elapsed,
            }
            metrics.update(update_metrics)
            with self.metrics_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(metrics, sort_keys=True) + "\n")

            print(
                "[train] "
                f"update={update}/{num_updates} "
                f"env_steps={self.total_env_steps} "
                f"variant={self.args.reward_variant} "
                f"struct={self.structure_variant} "
                f"return={mean_return:.3f} "
                f"len={mean_length:.1f} "
                f"success={success_rate:.3f} "
                f"goals={mean_goals:.3f} "
                f"passes={mean_passes:.3f} "
                f"assists={mean_assists:.3f} "
                f"pass_to_shot={mean_pass_to_shot:.3f} "
                f"possession={mean_possession:.2f} "
                f"policy_loss={update_metrics['policy_loss']:.4f} "
                f"value_loss={update_metrics['value_loss']:.4f} "
                f"entropy={update_metrics['entropy']:.4f} "
                f"kl={update_metrics['approx_kl']:.5f}",
                flush=True,
            )

            if update % int(self.args.save_interval) == 0 or update == num_updates:
                latest_checkpoint = self._save_checkpoint(update)

        self.envs.close()
        return latest_checkpoint
