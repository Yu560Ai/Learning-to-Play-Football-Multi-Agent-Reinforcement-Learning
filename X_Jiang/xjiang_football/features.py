from __future__ import annotations

from typing import Any

import numpy as np

from xjiang_football.utils import (
    ROLE_GK,
    ROLE_ST,
    ControlledRoleMap,
    build_controlled_role_map,
    clip_unit,
    distance,
    normalize_distance,
    primary_observation,
    attacking_support_target,
    role_attack_position,
    role_home_position,
    role_one_hot,
    role_recover_position,
    role_support_position,
    safe_ball_direction,
    safe_ball_xy,
    safe_owned_player,
    safe_owned_team,
    safe_team_directions,
    safe_team_positions,
)


def feature_dim() -> int:
    return 41


def extract_feature_metrics(raw_observation: Any, role_map: ControlledRoleMap | None = None) -> dict[str, float]:
    default = {
        "goalkeeper_x": float("nan"),
        "active_player_distance_to_ball": float("nan"),
        "closest_outfield_distance_to_ball": float("nan"),
        "second_outfield_distance_to_ball": float("nan"),
        "team_spread": float("nan"),
        "progression_estimate": 0.0,
    }
    obs = primary_observation(raw_observation)
    if obs is None:
        return default
    if role_map is None:
        role_map = build_controlled_role_map(raw_observation, requested_players=5)
    if role_map.controlled_indices.size == 0:
        return default

    left_team = safe_team_positions(obs, "left_team")
    ball_xy = safe_ball_xy(obs)
    owned_player = safe_owned_player(obs)
    controlled_positions = left_team[role_map.controlled_indices]
    distances = np.linalg.norm(controlled_positions - ball_xy, axis=1)
    outfield_mask = role_map.role_ids != ROLE_GK
    outfield_distances = np.sort(distances[outfield_mask]) if np.any(outfield_mask) else np.asarray([], dtype=np.float32)
    active_distance = float("nan")
    if owned_player in role_map.controlled_indices.tolist():
        slot = int(np.where(role_map.controlled_indices == owned_player)[0][0])
        active_distance = float(distances[slot])
    elif distances.size > 0:
        active_distance = float(np.min(distances))

    goalkeeper_x = float("nan")
    if np.any(role_map.role_ids == ROLE_GK):
        goalkeeper_x = float(np.mean(controlled_positions[role_map.role_ids == ROLE_GK, 0]))

    team_spread = float(np.mean(np.std(controlled_positions, axis=0))) if controlled_positions.shape[0] > 1 else 0.0
    return {
        "goalkeeper_x": goalkeeper_x,
        "active_player_distance_to_ball": active_distance,
        "closest_outfield_distance_to_ball": float(outfield_distances[0]) if outfield_distances.size > 0 else float("nan"),
        "second_outfield_distance_to_ball": float(outfield_distances[1]) if outfield_distances.size > 1 else float("nan"),
        "team_spread": team_spread,
        "progression_estimate": float(np.clip((float(ball_xy[0]) + 1.0) / 2.0, 0.0, 1.0)),
    }


def build_tactical_features(raw_observation: Any, role_map: ControlledRoleMap | None = None) -> np.ndarray:
    obs = primary_observation(raw_observation)
    if obs is None:
        return np.zeros((1, feature_dim()), dtype=np.float32)

    if role_map is None:
        role_map = build_controlled_role_map(raw_observation, requested_players=5)

    left_team = safe_team_positions(obs, "left_team")
    right_team = safe_team_positions(obs, "right_team")
    left_dir = safe_team_directions(obs, "left_team_direction", left_team.shape[0])
    ball_xy = safe_ball_xy(obs)
    ball_direction = safe_ball_direction(obs)
    owned_team = safe_owned_team(obs)
    owned_player = safe_owned_player(obs)
    own_team_has_ball = owned_team == 0
    opponent_has_ball = owned_team == 1

    if role_map.controlled_indices.size == 0:
        return np.zeros((1, feature_dim()), dtype=np.float32)

    controlled_positions = left_team[role_map.controlled_indices]
    controlled_ball_distances = np.linalg.norm(controlled_positions - ball_xy, axis=1)
    sorted_outfield_slots = np.argsort(
        np.where(role_map.role_ids == ROLE_GK, 1e9, controlled_ball_distances)
    )
    closest_slot = int(sorted_outfield_slots[0]) if sorted_outfield_slots.size > 0 else -1
    second_slot = int(sorted_outfield_slots[1]) if sorted_outfield_slots.size > 1 else -1

    features: list[np.ndarray] = []
    for slot, (actual_idx, role_id) in enumerate(zip(role_map.controlled_indices.tolist(), role_map.role_ids.tolist())):
        player_xy = left_team[actual_idx]
        player_dir = left_dir[actual_idx]
        rel_ball = ball_xy - player_xy
        ball_distance = float(np.linalg.norm(rel_ball))
        nearest_teammate = 0.0
        if left_team.shape[0] > 1:
            teammate_positions = np.delete(left_team, actual_idx, axis=0)
            nearest_teammate = float(np.min(np.linalg.norm(teammate_positions - player_xy, axis=1)))
        nearest_opponent = 0.0
        if right_team.shape[0] > 0:
            nearest_opponent = float(np.min(np.linalg.norm(right_team - player_xy, axis=1)))

        home_target = role_home_position(role_id, ball_xy, own_team_has_ball)
        support_target = attacking_support_target(obs, actual_idx, role_id, aggressive=False) if own_team_has_ball else role_support_position(role_id, ball_xy, own_team_has_ball)
        recover_target = role_recover_position(role_id, ball_xy)
        attack_target = attacking_support_target(obs, actual_idx, role_id, aggressive=True) if own_team_has_ball else role_attack_position(role_id, ball_xy)

        vec = np.concatenate(
            [
                role_one_hot(role_id),
                np.asarray(
                    [
                        clip_unit(float(player_xy[0])),
                        clip_unit(float(player_xy[1])),
                        clip_unit(float(player_dir[0]), limit=0.15),
                        clip_unit(float(player_dir[1]), limit=0.15),
                        clip_unit(float(rel_ball[0])),
                        clip_unit(float(rel_ball[1])),
                        normalize_distance(ball_distance),
                        clip_unit(float(ball_xy[0])),
                        clip_unit(float(ball_xy[1])),
                        clip_unit(float(ball_direction[0]), limit=0.2),
                        clip_unit(float(ball_direction[1]), limit=0.2),
                        1.0 if own_team_has_ball else 0.0,
                        1.0 if opponent_has_ball else 0.0,
                        1.0 if owned_team < 0 else 0.0,
                        1.0 if owned_player == actual_idx and own_team_has_ball else 0.0,
                        1.0 if slot == closest_slot else 0.0,
                        1.0 if slot == second_slot else 0.0,
                        normalize_distance(nearest_teammate),
                        normalize_distance(nearest_opponent),
                        clip_unit(float(home_target[0] - player_xy[0])),
                        clip_unit(float(home_target[1] - player_xy[1])),
                        normalize_distance(distance(home_target, player_xy)),
                        clip_unit(float(support_target[0] - player_xy[0])),
                        clip_unit(float(support_target[1] - player_xy[1])),
                        normalize_distance(distance(support_target, player_xy)),
                        clip_unit(float(recover_target[0] - player_xy[0])),
                        clip_unit(float(recover_target[1] - player_xy[1])),
                        normalize_distance(distance(recover_target, player_xy)),
                        clip_unit(float(attack_target[0] - player_xy[0])),
                        clip_unit(float(attack_target[1] - player_xy[1])),
                        normalize_distance(distance(attack_target, player_xy)),
                        1.0 if player_xy[1] < -0.18 else 0.0,
                        1.0 if -0.18 <= player_xy[1] <= 0.18 else 0.0,
                        1.0 if player_xy[1] > 0.18 else 0.0,
                        1.0 if player_xy[0] < -0.35 else 0.0,
                        1.0 if -0.35 <= player_xy[0] <= 0.35 else 0.0,
                        1.0 if player_xy[0] > 0.35 else 0.0,
                    ],
                    dtype=np.float32,
                ),
            ]
        )
        features.append(vec.astype(np.float32))

    return np.stack(features, axis=0)
