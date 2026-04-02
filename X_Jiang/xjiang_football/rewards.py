from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from xjiang_football.features import extract_feature_metrics
from xjiang_football.utils import (
    GOALKEEPER_ROLE,
    ROLE_GK,
    _best_forward_teammate,
    _best_reset_teammate,
    attacking_support_target,
    controlled_team_has_free_ball_advantage,
    TACTICAL_MODE_FIRST_DEFENDER,
    TACTICAL_MODE_ON_BALL,
    TACTICAL_MODE_SUPPORT_ATTACK,
    TACTICAL_MODE_SUPPORT_COVER,
    TACTICAL_ATTACK_SPACE,
    TACTICAL_CHASE_BALL,
    TACTICAL_CLEAR_BALL,
    TACTICAL_HOLD,
    TACTICAL_PROGRESSIVE_PASS,
    TACTICAL_RECOVER,
    TACTICAL_SAFE_RESET_PASS,
    TACTICAL_SHOOT_BALL,
    TACTICAL_SUPPORT_BALL,
    ControlledRoleMap,
    choose_relevant_target,
    distance,
    inferred_controlled_ball_owner_slot,
    nearest_opponent_distance_to_point,
    primary_observation,
    safe_ball_xy,
    safe_owned_player,
    safe_roles,
    safe_owned_team,
    safe_team_positions,
    tactical_mode_indices,
)


@dataclass(frozen=True)
class RewardShapingConfig:
    enabled: bool = True
    closest_player_to_ball_reward: float = 0.08
    first_defender_pressure_reward: float = 0.06
    second_player_support_reward: float = 0.05
    recover_shape_reward: float = 0.04
    hold_shape_reward: float = 0.02
    ball_watch_penalty: float = 0.06
    idle_wander_penalty: float = 0.03
    goalkeeper_home_reward: float = 0.03
    goalkeeper_wander_penalty: float = 0.08
    possession_support_reward: float = 0.03
    attack_space_reward: float = 0.04
    progressive_pass_choice_reward: float = 0.05
    progressive_pass_result_reward_scale: float = 0.10
    carry_progress_reward_scale: float = 0.06
    zone_entry_progress_reward: float = 0.10
    safe_reset_pass_reward: float = 0.01
    backward_gk_pass_penalty: float = 0.08
    unnecessary_goalkeeper_reset_penalty: float = 0.12
    non_emergency_clear_penalty: float = 0.10
    shot_choice_reward: float = 0.05
    missed_shot_window_penalty: float = 0.10
    shot_execution_reward: float = 0.15
    on_ball_stall_penalty: float = 0.08
    on_ball_backward_drift_penalty: float = 0.07
    on_ball_lateral_zigzag_penalty: float = 0.05
    support_spacing_reward: float = 0.05
    support_spacing_penalty: float = 0.04
    support_forward_lane_reward: float = 0.04
    support_static_penalty: float = 0.03
    safe_reset_overuse_penalty: float = 0.10
    support_behind_ball_penalty: float = 0.05


def _empty_reward(role_map: ControlledRoleMap) -> tuple[np.ndarray, dict[str, float]]:
    return np.zeros(role_map.role_ids.size, dtype=np.float32), {}


def extract_step_metrics(raw_observation: Any, role_map: ControlledRoleMap | None = None) -> dict[str, float]:
    return extract_feature_metrics(raw_observation, role_map=role_map)


def compute_shaped_reward(
    previous_raw_observation: Any,
    next_raw_observation: Any,
    tactical_actions: np.ndarray,
    role_map: ControlledRoleMap,
    config: RewardShapingConfig,
) -> tuple[np.ndarray, dict[str, float]]:
    if not config.enabled or role_map.role_ids.size == 0:
        return _empty_reward(role_map)

    previous = primary_observation(previous_raw_observation)
    current = primary_observation(next_raw_observation)
    if previous is None or current is None:
        return _empty_reward(role_map)

    prev_team = safe_team_positions(previous, "left_team")
    current_team = safe_team_positions(current, "left_team")
    if prev_team.shape[0] == 0 or current_team.shape[0] == 0:
        return _empty_reward(role_map)

    controlled_indices = role_map.controlled_indices
    role_ids = role_map.role_ids
    prev_positions = prev_team[controlled_indices]
    current_positions = current_team[controlled_indices]
    prev_ball_xy = safe_ball_xy(previous)
    current_ball_xy = safe_ball_xy(current)
    prev_owned_team = safe_owned_team(previous)
    current_owned_team = safe_owned_team(current)
    current_owned_player = safe_owned_player(current)
    prev_owner_slot = inferred_controlled_ball_owner_slot(previous, role_map)
    current_owner_slot = inferred_controlled_ball_owner_slot(current, role_map)
    current_modes = tactical_mode_indices(current, role_map)
    prev_roles = safe_roles(previous, prev_team.shape[0])
    goalkeeper_candidates = np.flatnonzero(prev_roles == GOALKEEPER_ROLE)
    goalkeeper_index = int(goalkeeper_candidates[0]) if goalkeeper_candidates.size > 0 else -1

    rewards = np.zeros(role_ids.size, dtype=np.float32)
    own_team_has_ball = current_owned_team == 0 or current_owner_slot >= 0
    pass_target_spaces: list[float] = []
    missed_shot_windows = 0.0
    gk_reset_events = 0.0
    free_ball_attack_phase = 1.0 if controlled_team_has_free_ball_advantage(current, role_map) else 0.0

    for slot, role_id in enumerate(role_ids.tolist()):
        player_xy = current_positions[slot]
        prev_xy = prev_positions[slot]
        action_id = int(tactical_actions[slot])
        mode_id = int(current_modes[slot]) if current_modes.size > slot else TACTICAL_MODE_SUPPORT_COVER
        if role_id == ROLE_GK:
            gk_home = choose_relevant_target(role_id, TACTICAL_HOLD, current_ball_xy, own_team_has_ball)
            home_dist = distance(player_xy, gk_home)
            rewards[slot] += max(0.0, 0.25 - home_dist) * config.goalkeeper_home_reward
            if player_xy[0] > -0.55:
                rewards[slot] -= (player_xy[0] + 0.55) * config.goalkeeper_wander_penalty
            continue

        if mode_id == TACTICAL_MODE_FIRST_DEFENDER:
            prev_d = float(np.linalg.norm(prev_xy - prev_ball_xy))
            curr_d = float(np.linalg.norm(player_xy - current_ball_xy))
            improvement = prev_d - curr_d
            if action_id == TACTICAL_CHASE_BALL:
                rewards[slot] += improvement * config.closest_player_to_ball_reward
                rewards[slot] += max(0.0, 0.18 - curr_d) * config.first_defender_pressure_reward
            else:
                rewards[slot] -= 0.5 * config.ball_watch_penalty
            if improvement < 0.0:
                rewards[slot] += improvement * config.ball_watch_penalty
            if curr_d > 0.22:
                rewards[slot] -= 0.5 * config.first_defender_pressure_reward

        if mode_id == TACTICAL_MODE_SUPPORT_COVER:
            recover_target = choose_relevant_target(role_id, TACTICAL_RECOVER, current_ball_xy, False)
            support_target = choose_relevant_target(role_id, TACTICAL_SUPPORT_BALL, current_ball_xy, False)
            prev_recover_dist = distance(prev_xy, recover_target)
            curr_recover_dist = distance(player_xy, recover_target)
            prev_support_dist = distance(prev_xy, support_target)
            curr_support_dist = distance(player_xy, support_target)
            if action_id == TACTICAL_RECOVER:
                rewards[slot] += (prev_recover_dist - curr_recover_dist) * config.recover_shape_reward
            elif action_id == TACTICAL_SUPPORT_BALL:
                rewards[slot] += (prev_support_dist - curr_support_dist) * config.second_player_support_reward
            else:
                rewards[slot] -= 0.25 * config.ball_watch_penalty

        if mode_id == TACTICAL_MODE_SUPPORT_ATTACK and current_owner_slot != slot:
            owner_slots = np.where(current_modes == TACTICAL_MODE_ON_BALL)[0]
            if owner_slots.size > 0:
                owner_slot = int(owner_slots[0])
                carrier_xy = current_positions[owner_slot]
                prev_carrier_xy = prev_positions[owner_slot]
                support_dist = distance(player_xy, carrier_xy)
                if 0.12 <= support_dist <= 0.32:
                    rewards[slot] += config.support_spacing_reward
                elif support_dist > 0.42 or support_dist < 0.08:
                    rewards[slot] -= config.support_spacing_penalty
                rel_x = float(player_xy[0] - carrier_xy[0])
                rel_y = abs(float(player_xy[1] - carrier_xy[1]))
                if 0.03 <= rel_x <= 0.28 and rel_y <= 0.24:
                    rewards[slot] += config.support_forward_lane_reward
                elif rel_x < -0.03:
                    rewards[slot] -= config.support_behind_ball_penalty
                if action_id == TACTICAL_ATTACK_SPACE and rel_x > 0.06 and rel_y <= 0.28:
                    rewards[slot] += 0.75 * config.support_forward_lane_reward
                if action_id == TACTICAL_SUPPORT_BALL and rel_x < 0.0:
                    rewards[slot] -= 0.5 * config.support_behind_ball_penalty
                if distance(prev_xy, player_xy) < 0.01 and distance(prev_carrier_xy, carrier_xy) < 0.015:
                    rewards[slot] -= config.support_static_penalty
            if action_id == TACTICAL_SUPPORT_BALL:
                support_target = attacking_support_target(current, int(controlled_indices[slot]), role_id, aggressive=False)
                rewards[slot] += (distance(prev_xy, support_target) - distance(player_xy, support_target)) * config.possession_support_reward
            if action_id == TACTICAL_ATTACK_SPACE:
                attack_target = attacking_support_target(current, int(controlled_indices[slot]), role_id, aggressive=True)
                rewards[slot] += (distance(prev_xy, attack_target) - distance(player_xy, attack_target)) * config.attack_space_reward

        hold_target = choose_relevant_target(role_id, TACTICAL_HOLD, current_ball_xy, own_team_has_ball)
        hold_dist = distance(player_xy, hold_target)
        if action_id == TACTICAL_HOLD:
            rewards[slot] += max(0.0, 0.35 - hold_dist) * config.hold_shape_reward
        elif hold_dist > 0.75:
            rewards[slot] -= (hold_dist - 0.75) * config.idle_wander_penalty

        if mode_id == TACTICAL_MODE_ON_BALL and own_team_has_ball and current_owner_slot == slot:
            ball_progress = float(current_ball_xy[0] - prev_ball_xy[0])
            zone_progress = max(0.0, float(np.floor((current_ball_xy[0] + 1.0) / 0.33) - np.floor((prev_ball_xy[0] + 1.0) / 0.33)))
            same_owner = prev_owner_slot == slot and current_owner_slot == slot
            actual_idx = int(controlled_indices[slot])
            if action_id == TACTICAL_PROGRESSIVE_PASS:
                teammate_idx, _ = _best_forward_teammate(
                    prev_team,
                    actual_idx,
                    previous,
                    min_forward_x=0.04,
                    allow_goalkeeper=False,
                    goalkeeper_index=goalkeeper_index if goalkeeper_index >= 0 else None,
                )
                if teammate_idx is not None:
                    pass_target_spaces.append(nearest_opponent_distance_to_point(previous, prev_team[teammate_idx]))
                rewards[slot] += config.progressive_pass_choice_reward
                rewards[slot] += max(0.0, ball_progress) * config.progressive_pass_result_reward_scale
            elif action_id == TACTICAL_SAFE_RESET_PASS:
                allow_goalkeeper = role_map.goalkeeper_controlled and (
                    role_id == ROLE_GK
                    or float(prev_ball_xy[0]) < -0.45
                    or (distance(prev_xy, prev_ball_xy) < 0.08 and float(prev_ball_xy[0]) < -0.2)
                )
                teammate_idx, _ = _best_reset_teammate(
                    prev_team,
                    actual_idx,
                    goalkeeper_index=goalkeeper_index if goalkeeper_index >= 0 else None,
                    allow_goalkeeper=allow_goalkeeper,
                    danger_ball_x=float(prev_ball_xy[0]),
                )
                if teammate_idx is not None:
                    pass_target_spaces.append(nearest_opponent_distance_to_point(previous, prev_team[teammate_idx]))
                    if goalkeeper_index >= 0 and teammate_idx == goalkeeper_index:
                        gk_reset_events += 1.0
                rewards[slot] += config.safe_reset_pass_reward
                if role_id != ROLE_GK and prev_ball_xy[0] > -0.18 and nearest_opponent_distance_to_point(previous, prev_xy) > 0.12:
                    rewards[slot] -= config.safe_reset_overuse_penalty
                if role_id != ROLE_GK and prev_ball_xy[0] > 0.0:
                    rewards[slot] -= 0.75 * config.safe_reset_overuse_penalty
                if role_id != ROLE_GK and current_ball_xy[0] < prev_ball_xy[0] - 0.05:
                    rewards[slot] -= config.backward_gk_pass_penalty
                if (
                    role_id != ROLE_GK
                    and prev_ball_xy[0] > -0.35
                    and current_ball_xy[0] < -0.65
                    and abs(float(current_ball_xy[1])) < 0.18
                ):
                    rewards[slot] -= config.unnecessary_goalkeeper_reset_penalty
            elif action_id == TACTICAL_SHOOT_BALL and float(current_ball_xy[0]) > 0.18:
                rewards[slot] += config.shot_choice_reward
                rewards[slot] += config.shot_execution_reward
            elif action_id == TACTICAL_CLEAR_BALL and float(current_ball_xy[0]) < -0.1:
                rewards[slot] += 0.5 * config.progressive_pass_choice_reward
                if role_id != ROLE_GK and prev_ball_xy[0] > -0.28:
                    rewards[slot] -= config.non_emergency_clear_penalty

            if (
                float(prev_ball_xy[0]) > 0.18
                and abs(float(prev_ball_xy[1])) < 0.34
                and action_id != TACTICAL_SHOOT_BALL
            ):
                missed_shot_windows += 1.0
                rewards[slot] -= config.missed_shot_window_penalty

            if same_owner:
                rewards[slot] += max(0.0, ball_progress) * config.carry_progress_reward_scale
                rewards[slot] += zone_progress * config.zone_entry_progress_reward
                if ball_progress < 0.015 and distance(prev_xy, player_xy) < 0.012:
                    rewards[slot] -= config.on_ball_stall_penalty
                if ball_progress < -0.01:
                    rewards[slot] -= config.on_ball_backward_drift_penalty
                if abs(float(current_ball_xy[1] - prev_ball_xy[1])) > 0.08 and ball_progress < 0.02:
                    rewards[slot] -= config.on_ball_lateral_zigzag_penalty
            elif action_id == TACTICAL_PROGRESSIVE_PASS and ball_progress <= 0.0:
                rewards[slot] -= 0.5 * config.progressive_pass_choice_reward

    info = extract_feature_metrics(next_raw_observation, role_map=role_map)
    info["reward_bonus"] = float(np.mean(rewards)) if rewards.size > 0 else 0.0
    info["pass_target_space"] = float(np.mean(pass_target_spaces)) if pass_target_spaces else float("nan")
    info["missed_shot_windows"] = float(missed_shot_windows)
    info["gk_reset_events"] = float(gk_reset_events)
    info["free_ball_attack_phase"] = float(free_ball_attack_phase)
    return rewards.astype(np.float32), info
