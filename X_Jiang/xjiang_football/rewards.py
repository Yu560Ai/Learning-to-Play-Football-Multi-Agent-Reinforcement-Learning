from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

@dataclass(frozen=True)
class RewardShapingConfig:
    enabled: bool = True
    possession_gain_reward: float = 0.15
    possession_loss_penalty: float = 0.15
    team_possession_reward: float = 0.001
    opponent_possession_penalty: float = 0.001
    successful_pass_reward: float = 0.03
    progressive_pass_reward_scale: float = 0.08
    carry_progress_reward_scale: float = 0.05
    attacking_third_reward: float = 0.002
    shot_with_ball_reward: float = 0.005
    checkpoint_reward_scale: float = 0.03
    attacking_risk_x_threshold: float = 0.4
    shot_reward_x_threshold: float = 0.35
    danger_zone_x_threshold: float = 0.2
    danger_zone_entry_reward: float = 0.015
    terminal_zone_x_threshold: float = 0.55
    terminal_zone_reward: float = 0.02
    finish_quality_threshold: float = 0.35
    finish_quality_progress_reward_scale: float = 0.04
    duel_reward_scale: float = 0.06
    low_quality_shot_penalty_scale: float = 0.04
    backtracking_penalty_scale: float = 0.04
    danger_zone_stall_penalty: float = 0.01
    bad_shot_penalty: float = 0.02
    attacking_loss_penalty_scale: float = 0.6
    out_of_play_loss_penalty: float = 0.05


CHECKPOINT_XS: tuple[float, ...] = (-0.35, -0.1, 0.15, 0.35, 0.5, 0.62, 0.72)
CHECKPOINT_WEIGHTS: tuple[float, ...] = (0.25, 0.35, 0.5, 0.75, 1.1, 1.6, 2.2)


def _raw_list(raw_observation: Any) -> list[dict[str, Any]]:
    if isinstance(raw_observation, list):
        return [item for item in raw_observation if isinstance(item, dict)]
    if isinstance(raw_observation, dict):
        return [raw_observation]
    return []


def _safe_ball_x(raw_obs: dict[str, Any]) -> float:
    ball = np.asarray(raw_obs.get("ball", np.zeros(3)), dtype=np.float32)
    return float(ball[0]) if ball.size > 0 else 0.0


def _checkpoint_index(ball_x: float) -> int:
    index = 0
    for threshold in CHECKPOINT_XS:
        if ball_x >= threshold:
            index += 1
    return index


def _checkpoint_value(ball_x: float) -> float:
    value = 0.0
    for threshold, weight in zip(CHECKPOINT_XS, CHECKPOINT_WEIGHTS):
        if ball_x >= threshold:
            value += weight
    return value


def _finish_quality(raw_obs: dict[str, Any], *, danger_zone_x_threshold: float) -> float:
    ball = np.asarray(raw_obs.get("ball", np.zeros(3)), dtype=np.float32)
    ball_x = float(ball[0]) if ball.size > 0 else 0.0
    ball_y = float(ball[1]) if ball.size > 1 else 0.0
    right_team = np.asarray(raw_obs.get("right_team", np.zeros((0, 2))), dtype=np.float32)
    goalkeeper_y = float(right_team[0][1]) if right_team.ndim == 2 and right_team.shape[0] > 0 and right_team.shape[1] >= 2 else 0.0

    x_progress = np.clip((ball_x - danger_zone_x_threshold) / max(1e-6, 1.0 - danger_zone_x_threshold), 0.0, 1.0)
    distance_to_goal = float(np.hypot(1.0 - ball_x, ball_y))
    distance_score = np.clip(1.0 - distance_to_goal / 1.15, 0.0, 1.0)
    centrality = np.clip(1.0 - abs(ball_y) / 0.42, 0.0, 1.0)
    goalkeeper_block = np.clip(1.0 - abs(ball_y - goalkeeper_y) / 0.30, 0.0, 1.0)

    quality = 0.40 * x_progress + 0.35 * distance_score + 0.25 * centrality
    quality *= 1.0 - 0.35 * goalkeeper_block
    return float(np.clip(quality, 0.0, 1.0))


def _nearest_outfield_defender_delta(raw_obs: dict[str, Any], player_xy: np.ndarray) -> np.ndarray | None:
    right_team = np.asarray(raw_obs.get("right_team", np.zeros((0, 2))), dtype=np.float32)
    if right_team.ndim != 2 or right_team.shape[1] < 2:
        return None
    # Treat index 0 as goalkeeper when present. In academy_empty_goal* drills the
    # opponent may only have a goalkeeper, which should not count as an outfield
    # defender for duel statistics/rewards.
    candidates = right_team[1:] if right_team.shape[0] > 1 else np.zeros((0, 2), dtype=np.float32)
    if candidates.shape[0] == 0:
        return None
    deltas = candidates[:, :2] - player_xy.reshape(1, 2)
    distances = np.linalg.norm(deltas, axis=1)
    return deltas[int(np.argmin(distances))].astype(np.float32)


def _duel_state(raw_obs: dict[str, Any], *, player_xy: np.ndarray, ball_x: float) -> tuple[float, float]:
    delta = _nearest_outfield_defender_delta(raw_obs, player_xy)
    if delta is None:
        return 0.0, 0.0
    dx = float(delta[0])
    dy = float(delta[1])
    distance = float(np.linalg.norm(delta))
    engaged = 1.0 if (ball_x > 0.0 and 0.0 <= dx <= 0.28 and abs(dy) <= 0.18 and distance <= 0.32) else 0.0
    return engaged, distance


def _use_minimal_progression_mode(config: RewardShapingConfig) -> bool:
    return (
        config.successful_pass_reward == 0.0
        and config.progressive_pass_reward_scale == 0.0
        and config.carry_progress_reward_scale == 0.0
        and config.attacking_third_reward == 0.0
        and config.shot_with_ball_reward == 0.0
        and config.danger_zone_entry_reward == 0.0
        and config.terminal_zone_reward == 0.0
        and config.finish_quality_progress_reward_scale == 0.0
        and config.duel_reward_scale == 0.0
        and config.low_quality_shot_penalty_scale == 0.0
        and config.backtracking_penalty_scale == 0.0
        and config.danger_zone_stall_penalty == 0.0
        and config.bad_shot_penalty == 0.0
    )

def compute_shaped_reward(
    previous_raw_observation: Any,
    next_raw_observation: Any,
    mapped_action: np.ndarray,
    config: RewardShapingConfig,
    num_players: int,
    checkpoint_progress_override: float = 0.0,
) -> tuple[np.ndarray, dict[str, float]]:
    rewards = np.zeros(num_players, dtype=np.float32)
    metrics_total = {
        "possession_gains": 0.0,
        "possession_losses": 0.0,
        "team_possession_steps": 0.0,
        "opponent_possession_steps": 0.0,
        "successful_passes": 0.0,
        "forward_ball_progress": 0.0,
        "carry_progress": 0.0,
        "attacking_third_possession_steps": 0.0,
        "danger_zone_entries": 0.0,
        "terminal_zone_entries": 0.0,
        "duel_engagements": 0.0,
        "duel_beats": 0.0,
        "shot_actions": 0.0,
        "shots_with_ball": 0.0,
        "ball_out_losses": 0.0,
        "checkpoint_progress": 0.0,
        "shared_reward": 0.0,
        "individual_reward": 0.0,
        "possession_reward": 0.0,
        "checkpoint_reward": 0.0,
        "pass_reward": 0.0,
        "carry_reward": 0.0,
        "territory_reward": 0.0,
        "shot_reward": 0.0,
        "backtracking_penalty": 0.0,
        "stall_penalty": 0.0,
        "bad_shot_penalty": 0.0,
        "out_penalty": 0.0,
        "danger_zone_reward": 0.0,
        "terminal_zone_reward": 0.0,
        "finish_quality": 0.0,
        "finish_quality_progress": 0.0,
        "finish_quality_progress_reward": 0.0,
        "duel_reward": 0.0,
        "low_quality_shot_penalty": 0.0,
        "reward_bonus": 0.0,
    }
    if not config.enabled or num_players <= 0:
        return rewards, metrics_total

    prev_list = _raw_list(previous_raw_observation)
    next_list = _raw_list(next_raw_observation)
    if not prev_list or not next_list:
        return rewards, metrics_total

    prev_team = int(prev_list[0].get("ball_owned_team", -1))
    next_team = int(next_list[0].get("ball_owned_team", -1))
    prev_ball_x = _safe_ball_x(prev_list[0])
    next_ball_x = _safe_ball_x(next_list[0])
    checkpoint_gain = max(0.0, float(checkpoint_progress_override))
    next_game_mode = int(next_list[0].get("game_mode", 0))
    attacking_third = 1.0 if next_team == 0 and next_ball_x >= 0.5 else 0.0
    danger_zone_entry = 1.0 if (
        next_team == 0
        and next_ball_x >= config.danger_zone_x_threshold
        and (prev_team != 0 or prev_ball_x < config.danger_zone_x_threshold)
    ) else 0.0
    terminal_zone_entry = 1.0 if (
        next_team == 0
        and next_ball_x >= config.terminal_zone_x_threshold
        and (prev_team != 0 or prev_ball_x < config.terminal_zone_x_threshold)
    ) else 0.0
    attacking_loss_scale = (
        config.attacking_loss_penalty_scale
        if prev_team == 0 and prev_ball_x >= config.attacking_risk_x_threshold
        else 1.0
    )
    out_of_play_loss = 1.0 if prev_team == 0 and next_team == -1 and next_game_mode in (2, 5) else 0.0

    shared_reward = 0.0
    if prev_team != 0 and next_team == 0:
        shared_reward += config.possession_gain_reward
    if prev_team == 0 and next_team != 0:
        shared_reward -= config.possession_loss_penalty * attacking_loss_scale
    if next_team == 0:
        shared_reward += config.team_possession_reward
    elif next_team == 1:
        shared_reward -= config.opponent_possession_penalty
    checkpoint_reward = checkpoint_gain * config.checkpoint_reward_scale
    shared_reward += checkpoint_reward
    shared_reward += attacking_third * config.attacking_third_reward
    danger_zone_reward = danger_zone_entry * config.danger_zone_entry_reward
    terminal_zone_reward = terminal_zone_entry * config.terminal_zone_reward
    shared_reward += danger_zone_reward
    shared_reward += terminal_zone_reward
    shared_reward -= out_of_play_loss * config.out_of_play_loss_penalty

    rewards += shared_reward

    metrics_total["possession_gains"] = 1.0 if prev_team != 0 and next_team == 0 else 0.0
    metrics_total["possession_losses"] = 1.0 if prev_team == 0 and next_team != 0 else 0.0
    metrics_total["team_possession_steps"] = 1.0 if next_team == 0 else 0.0
    metrics_total["opponent_possession_steps"] = 1.0 if next_team == 1 else 0.0
    metrics_total["forward_ball_progress"] = max(0.0, next_ball_x - prev_ball_x)
    metrics_total["checkpoint_progress"] = float(checkpoint_gain)
    metrics_total["attacking_third_possession_steps"] = attacking_third
    metrics_total["danger_zone_entries"] = danger_zone_entry
    metrics_total["terminal_zone_entries"] = terminal_zone_entry
    metrics_total["ball_out_losses"] = out_of_play_loss
    metrics_total["shared_reward"] = float(shared_reward)
    metrics_total["danger_zone_reward"] = float(danger_zone_reward)
    metrics_total["terminal_zone_reward"] = float(terminal_zone_reward)
    metrics_total["possession_reward"] = float(
        (1.0 if prev_team != 0 and next_team == 0 else 0.0) * config.possession_gain_reward
        - (1.0 if prev_team == 0 and next_team != 0 else 0.0) * config.possession_loss_penalty * attacking_loss_scale
        + (1.0 if next_team == 0 else 0.0) * config.team_possession_reward
        - (1.0 if next_team == 1 else 0.0) * config.opponent_possession_penalty
    )
    metrics_total["checkpoint_reward"] = float(checkpoint_reward)

    if _use_minimal_progression_mode(config):
        for idx in range(min(num_players, len(prev_list), len(next_list))):
            previous = prev_list[idx]
            prev_team_local = int(previous.get("ball_owned_team", -1))
            action = int(mapped_action[idx]) if idx < mapped_action.shape[0] else 0
            shot_action = 1.0 if action == 12 else 0.0
            shot_in_good_zone = 1.0 if (
                shot_action > 0.0
                and prev_team_local == 0
                and _safe_ball_x(previous) >= config.shot_reward_x_threshold
            ) else 0.0
            metrics_total["shot_actions"] += shot_action
            metrics_total["shots_with_ball"] += shot_in_good_zone
            metrics_total["finish_quality"] += 0.0
        metrics_total["individual_reward"] = 0.0
        metrics_total["out_penalty"] = float(out_of_play_loss * config.out_of_play_loss_penalty)
        metrics_total["reward_bonus"] = float(np.mean(rewards)) if rewards.size > 0 else 0.0
        return rewards.astype(np.float32), metrics_total

    individual_reward_total = 0.0
    for idx in range(min(num_players, len(prev_list), len(next_list))):
        previous = prev_list[idx]
        current = next_list[idx]
        prev_player = int(previous.get("ball_owned_player", -1))
        next_player = int(current.get("ball_owned_player", -1))
        prev_team_local = int(previous.get("ball_owned_team", -1))
        next_team_local = int(current.get("ball_owned_team", -1))
        prev_ball_x_local = _safe_ball_x(previous)
        next_ball_x_local = _safe_ball_x(current)
        prev_left_team = np.asarray(previous.get("left_team", np.zeros((0, 2))), dtype=np.float32)
        next_left_team = np.asarray(current.get("left_team", np.zeros((0, 2))), dtype=np.float32)
        active_prev = int(previous.get("active", 0))
        active_next = int(current.get("active", active_prev))
        prev_player_xy = (
            prev_left_team[active_prev, :2]
            if prev_left_team.ndim == 2 and prev_left_team.shape[0] > active_prev and prev_left_team.shape[1] >= 2
            else np.zeros(2, dtype=np.float32)
        )
        next_player_xy = (
            next_left_team[active_next, :2]
            if next_left_team.ndim == 2 and next_left_team.shape[0] > active_next and next_left_team.shape[1] >= 2
            else np.zeros(2, dtype=np.float32)
        )

        action = int(mapped_action[idx]) if idx < mapped_action.shape[0] else 0
        pass_completed = 1.0 if (
            action in (9, 10, 11)
            and prev_team_local == 0
            and next_team_local == 0
            and next_player >= 0
            and next_player != prev_player
        ) else 0.0
        forward_ball_progress = max(0.0, next_ball_x_local - prev_ball_x_local)
        same_player_carry = 1.0 if (
            prev_team_local == 0 and next_team_local == 0 and prev_player >= 0 and prev_player == next_player
        ) else 0.0
        carry_progress = forward_ball_progress * same_player_carry
        shot_in_good_zone = prev_ball_x_local >= config.shot_reward_x_threshold
        prev_finish_quality = _finish_quality(previous, danger_zone_x_threshold=config.danger_zone_x_threshold)
        next_finish_quality = _finish_quality(current, danger_zone_x_threshold=config.danger_zone_x_threshold)
        finish_quality = prev_finish_quality
        duel_engaged, prev_defender_distance = _duel_state(previous, player_xy=prev_player_xy, ball_x=prev_ball_x_local)
        next_duel_engaged, next_defender_distance = _duel_state(current, player_xy=next_player_xy, ball_x=next_ball_x_local)
        shot_with_ball = 1.0 if action == 12 and prev_team_local == 0 and shot_in_good_zone else 0.0
        bad_shot = 1.0 if action == 12 and prev_team_local == 0 and not shot_in_good_zone else 0.0
        shot_action = 1.0 if action == 12 else 0.0
        backtracking = 1.0 if (
            same_player_carry > 0.0
            and prev_ball_x_local >= config.danger_zone_x_threshold
            and next_ball_x_local < prev_ball_x_local - 0.03
        ) else 0.0
        danger_zone_stall = 1.0 if (
            same_player_carry > 0.0
            and prev_ball_x_local >= config.danger_zone_x_threshold
            and abs(next_ball_x_local - prev_ball_x_local) < 0.005
            and action != 12
        ) else 0.0
        finish_quality_progress = max(0.0, next_finish_quality - prev_finish_quality) if same_player_carry > 0.0 else 0.0
        duel_success = 1.0 if (
            same_player_carry > 0.0
            and duel_engaged > 0.0
            and next_ball_x_local > prev_ball_x_local + 0.03
            and next_finish_quality > prev_finish_quality + 0.03
            and (next_duel_engaged <= 0.0 or next_defender_distance > prev_defender_distance + 0.05)
        ) else 0.0

        individual_reward = 0.0
        pass_reward = 0.0
        if pass_completed:
            pass_reward += config.successful_pass_reward
            pass_reward += forward_ball_progress * config.progressive_pass_reward_scale
            individual_reward += pass_reward
        carry_reward = carry_progress * config.carry_progress_reward_scale
        finish_quality_progress_reward = finish_quality_progress * config.finish_quality_progress_reward_scale
        duel_reward = duel_success * config.duel_reward_scale
        shot_reward = shot_with_ball * config.shot_with_ball_reward * max(
            0.0,
            (finish_quality - config.finish_quality_threshold) / max(1e-6, 1.0 - config.finish_quality_threshold),
        )
        bad_shot_penalty = bad_shot * config.bad_shot_penalty
        low_quality_shot_penalty = 0.0
        if shot_action > 0.0 and prev_team_local == 0 and finish_quality < config.finish_quality_threshold:
            low_quality_shot_penalty = (config.finish_quality_threshold - finish_quality) * config.low_quality_shot_penalty_scale
        backtracking_penalty = backtracking * config.backtracking_penalty_scale
        stall_penalty = danger_zone_stall * config.danger_zone_stall_penalty
        territory_reward = 0.0
        if idx == 0:
            territory_reward = danger_zone_reward + terminal_zone_reward + attacking_third * config.attacking_third_reward
        individual_reward += carry_reward
        individual_reward += finish_quality_progress_reward
        individual_reward += duel_reward
        individual_reward += shot_reward
        individual_reward -= backtracking_penalty
        individual_reward -= stall_penalty
        individual_reward -= bad_shot_penalty
        individual_reward -= low_quality_shot_penalty
        rewards[idx] += individual_reward
        individual_reward_total += float(individual_reward)

        metrics_total["successful_passes"] += pass_completed
        metrics_total["carry_progress"] += carry_progress
        metrics_total["duel_engagements"] += duel_engaged
        metrics_total["duel_beats"] += duel_success
        metrics_total["shot_actions"] += shot_action
        metrics_total["shots_with_ball"] += shot_with_ball
        metrics_total["pass_reward"] += float(pass_reward)
        metrics_total["carry_reward"] += float(carry_reward)
        metrics_total["finish_quality_progress"] += float(finish_quality_progress)
        metrics_total["finish_quality_progress_reward"] += float(finish_quality_progress_reward)
        metrics_total["duel_reward"] += float(duel_reward)
        metrics_total["shot_reward"] += float(shot_reward)
        metrics_total["finish_quality"] += float(finish_quality * shot_action)
        metrics_total["backtracking_penalty"] += float(backtracking_penalty)
        metrics_total["stall_penalty"] += float(stall_penalty)
        metrics_total["bad_shot_penalty"] += float(bad_shot_penalty)
        metrics_total["low_quality_shot_penalty"] += float(low_quality_shot_penalty)
        metrics_total["territory_reward"] += float(territory_reward)

    metrics_total["individual_reward"] = float(individual_reward_total)
    metrics_total["out_penalty"] = float(out_of_play_loss * config.out_of_play_loss_penalty)
    metrics_total["reward_bonus"] = float(np.mean(rewards)) if rewards.size > 0 else 0.0
    return rewards.astype(np.float32), metrics_total
