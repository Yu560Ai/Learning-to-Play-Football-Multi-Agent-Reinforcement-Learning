from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
import time

import numpy as np
import torch
import torch.nn.functional as F

from presets import build_config
from xjiang_football.envs import FootballEnvWrapper
from xjiang_football.model import ActorCritic, ModelConfig
from xjiang_football.priors import rule_based_single_player_action_from_obs
from xjiang_football.ppo import resolve_device, set_seed
from xjiang_football.rewards import RewardShapingConfig


def _format_duration(seconds: float) -> str:
    total = max(0, int(seconds))
    minutes, sec = divmod(total, 60)
    hours, minutes = divmod(minutes, 60)
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{sec:02d}"
    return f"{minutes:02d}:{sec:02d}"


def collect_rule_dataset(
    env: FootballEnvWrapper,
    target_samples: int,
    *,
    shoot_x_threshold: float,
    shoot_y_threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    observations: list[np.ndarray] = []
    actions: list[int] = []
    obs = env.reset()
    start_time = time.monotonic()
    report_interval = max(256, target_samples // 10)
    while len(actions) < target_samples:
        action = rule_based_single_player_action_from_obs(
            np.asarray(obs[0], dtype=np.float32),
            shoot_x_threshold=shoot_x_threshold,
            shoot_y_threshold=shoot_y_threshold,
        )
        observations.append(np.asarray(obs[0], dtype=np.float32))
        actions.append(int(action))
        collected = len(actions)
        if collected % report_interval == 0 or collected == target_samples:
            elapsed = time.monotonic() - start_time
            rate = collected / max(elapsed, 1e-6)
            remaining = (target_samples - collected) / max(rate, 1e-6)
            print(
                f"[collect] samples={collected}/{target_samples} "
                f"elapsed={_format_duration(elapsed)} remaining={_format_duration(remaining)}",
                flush=True,
            )
        obs, _, done, _ = env.step(np.asarray([action], dtype=np.int64))
        if done:
            obs = env.reset()
    return np.stack(observations, axis=0), np.asarray(actions, dtype=np.int64)


def pretrain_behavior_cloning(
    preset_name: str,
    output_path: str,
    samples: int,
    epochs: int,
    batch_size: int,
    lr: float,
    device: str,
) -> Path:
    config = build_config(preset_name)
    if config.num_controlled_players != 1:
        raise ValueError("Behavior pretraining currently expects a single-player preset.")

    resolved_device = resolve_device(device or config.device)
    set_seed(config.seed)
    reward_cfg = RewardShapingConfig(
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
    )
    env = FootballEnvWrapper(
        env_name=config.env_name,
        representation=config.representation,
        rewards=config.rewards,
        render=False,
        logdir=config.logdir,
        num_controlled_players=config.num_controlled_players,
        channel_dimensions=config.channel_dimensions,
        reward_shaping=reward_cfg,
        use_engineered_features=config.use_engineered_features,
        collect_feature_metrics=False,
        action_set=config.action_set,
        force_shoot_in_zone=config.force_shoot_in_zone,
        force_shoot_x_threshold=config.force_shoot_x_threshold,
        force_shoot_y_threshold=config.force_shoot_y_threshold,
    )
    print(f"[startup] collecting rule dataset samples={samples} preset={preset_name}", flush=True)
    obs_np, actions_np = collect_rule_dataset(
        env,
        samples,
        shoot_x_threshold=config.bc_shoot_x_threshold,
        shoot_y_threshold=config.bc_shoot_y_threshold,
    )
    env.close()

    action_dim = env.action_dim
    obs_dim = env.obs_dim
    obs_shape = env.obs_shape
    num_players = env.num_players
    action_names = env.action_names
    model = ActorCritic(
        obs_dim=obs_np.shape[1],
        action_dim=action_dim,
        num_players=1,
        config=ModelConfig(
            head_dim=config.head_dim,
            trunk_dim=config.trunk_dim,
            critic_dim=config.critic_dim,
        ),
    ).to(resolved_device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    obs_tensor = torch.as_tensor(obs_np, dtype=torch.float32, device=resolved_device)
    act_tensor = torch.as_tensor(actions_np, dtype=torch.int64, device=resolved_device)
    num_samples = obs_tensor.shape[0]

    print(
        f"[startup] behavior pretrain obs_dim={obs_tensor.shape[1]} action_dim={env.action_dim} "
        f"samples={num_samples} epochs={epochs} device={resolved_device}",
        flush=True,
    )
    for epoch in range(1, epochs + 1):
        permutation = torch.randperm(num_samples, device=resolved_device)
        losses: list[float] = []
        correct = 0
        for start in range(0, num_samples, batch_size):
            batch_idx = permutation[start : start + batch_size]
            logits, _ = model.actor_forward(obs_tensor[batch_idx])
            loss = F.cross_entropy(logits, act_tensor[batch_idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))
            correct += int((torch.argmax(logits, dim=-1) == act_tensor[batch_idx]).sum().item())
        accuracy = correct / float(num_samples)
        print(
            f"[pretrain {epoch}/{epochs}] loss={np.mean(losses):.4f} accuracy={accuracy:.3f}",
            flush=True,
        )

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": asdict(config),
            "obs_dim": obs_dim,
            "obs_shape": obs_shape,
            "action_dim": action_dim,
            "num_players": num_players,
            "action_names": action_names,
            "update": 0,
            "total_agent_steps": 0,
            "pretrain_samples": num_samples,
            "pretrain_epochs": epochs,
        },
        output,
    )
    print(f"[done] saved behavior prior to {output}", flush=True)
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Behavior cloning warm-start for single-player academy.")
    parser.add_argument("--preset", type=str, default="academy_run_to_score_attack_foundation")
    parser.add_argument(
        "--output",
        type=str,
        default="X_Jiang/checkpoints/academy_run_to_score_attack_foundation/behavior_prior.pt",
    )
    parser.add_argument("--samples", type=int, default=4096)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--device", type=str, default="cpu")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    pretrain_behavior_cloning(
        preset_name=args.preset,
        output_path=args.output,
        samples=args.samples,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.learning_rate,
        device=args.device,
    )
