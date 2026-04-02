from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


ROLE_GK = 0
ROLE_LD = 1
ROLE_RD = 2
ROLE_CM = 3
ROLE_ST = 4
NUM_ROLES = 5

ROLE_NAMES = {
    ROLE_GK: "gk",
    ROLE_LD: "ld",
    ROLE_RD: "rd",
    ROLE_CM: "cm",
    ROLE_ST: "st",
}

TACTICAL_HOLD = 0
TACTICAL_CHASE_BALL = 1
TACTICAL_SUPPORT_BALL = 2
TACTICAL_RECOVER = 3
TACTICAL_ATTACK_SPACE = 4
TACTICAL_PROGRESSIVE_PASS = 5
TACTICAL_SAFE_RESET_PASS = 6
TACTICAL_SHOOT_BALL = 7
TACTICAL_CLEAR_BALL = 8
NUM_TACTICAL_ACTIONS = 9

TACTICAL_MODE_GK = 0
TACTICAL_MODE_ON_BALL = 1
TACTICAL_MODE_FIRST_DEFENDER = 2
TACTICAL_MODE_SUPPORT_ATTACK = 3
TACTICAL_MODE_SUPPORT_COVER = 4
NUM_TACTICAL_MODES = 5

POLICY_HEAD_GK = TACTICAL_MODE_GK
POLICY_HEAD_ON_BALL = TACTICAL_MODE_ON_BALL
POLICY_HEAD_FIRST_DEFENDER = TACTICAL_MODE_FIRST_DEFENDER
POLICY_HEAD_SUPPORT_ATTACK = TACTICAL_MODE_SUPPORT_ATTACK
POLICY_HEAD_SUPPORT_COVER = TACTICAL_MODE_SUPPORT_COVER
NUM_POLICY_HEADS = NUM_TACTICAL_MODES

TACTICAL_ACTION_NAMES = {
    TACTICAL_HOLD: "hold_role",
    TACTICAL_CHASE_BALL: "chase_ball",
    TACTICAL_SUPPORT_BALL: "support_ball",
    TACTICAL_RECOVER: "recover_goal_side",
    TACTICAL_ATTACK_SPACE: "attack_space",
    TACTICAL_PROGRESSIVE_PASS: "progressive_pass",
    TACTICAL_SAFE_RESET_PASS: "safe_reset_pass",
    TACTICAL_SHOOT_BALL: "shoot_ball",
    TACTICAL_CLEAR_BALL: "clear_ball",
}

POLICY_HEAD_NAMES = {
    POLICY_HEAD_GK: "goalkeeper",
    POLICY_HEAD_ON_BALL: "on_ball",
    POLICY_HEAD_FIRST_DEFENDER: "first_defender",
    POLICY_HEAD_SUPPORT_ATTACK: "support_attack",
    POLICY_HEAD_SUPPORT_COVER: "support_cover",
}

TACTICAL_MODE_NAMES = {
    TACTICAL_MODE_GK: "goalkeeper",
    TACTICAL_MODE_ON_BALL: "on_ball",
    TACTICAL_MODE_FIRST_DEFENDER: "first_defender",
    TACTICAL_MODE_SUPPORT_ATTACK: "support_attack",
    TACTICAL_MODE_SUPPORT_COVER: "support_cover",
}

GOALKEEPER_ROLE = 0
OWN_GOAL = np.asarray([-1.0, 0.0], dtype=np.float32)
OPP_GOAL = np.asarray([1.0, 0.0], dtype=np.float32)

# Default GRF action set ids.
ACTION_IDLE = 0
ACTION_LEFT = 1
ACTION_TOP_LEFT = 2
ACTION_TOP = 3
ACTION_TOP_RIGHT = 4
ACTION_RIGHT = 5
ACTION_BOTTOM_RIGHT = 6
ACTION_BOTTOM = 7
ACTION_BOTTOM_LEFT = 8
ACTION_LONG_PASS = 9
ACTION_HIGH_PASS = 10
ACTION_SHORT_PASS = 11
ACTION_SHOT = 12
ACTION_SPRINT = 13
ACTION_RELEASE_DIRECTION = 14
ACTION_RELEASE_SPRINT = 15
ACTION_SLIDING = 16
ACTION_DRIBBLE = 17
ACTION_RELEASE_DRIBBLE = 18


@dataclass(frozen=True)
class ControlledRoleMap:
    controlled_indices: np.ndarray
    role_ids: np.ndarray
    role_names: tuple[str, ...]
    goalkeeper_controlled: bool


def observation_list(raw_observation: Any) -> list[dict[str, Any]]:
    if isinstance(raw_observation, list):
        return [item for item in raw_observation if isinstance(item, dict)]
    if isinstance(raw_observation, dict):
        return [raw_observation]
    return []


def primary_observation(raw_observation: Any) -> dict[str, Any] | None:
    items = observation_list(raw_observation)
    return items[0] if items else None


def safe_team_positions(raw_observation: dict[str, Any], key: str) -> np.ndarray:
    team = np.asarray(raw_observation.get(key, []), dtype=np.float32)
    if team.ndim != 2 or team.shape[0] == 0:
        return np.zeros((0, 2), dtype=np.float32)
    return team[:, :2]


def safe_team_directions(raw_observation: dict[str, Any], key: str, team_size: int) -> np.ndarray:
    directions = np.asarray(raw_observation.get(key, np.zeros((team_size, 2))), dtype=np.float32)
    if directions.ndim != 2 or directions.shape[0] != team_size:
        return np.zeros((team_size, 2), dtype=np.float32)
    return directions[:, :2]


def safe_roles(raw_observation: dict[str, Any], expected_size: int) -> np.ndarray:
    roles = np.asarray(raw_observation.get("left_team_roles", np.zeros(expected_size)), dtype=np.int64)
    if roles.ndim != 1 or roles.size != expected_size:
        return np.zeros(expected_size, dtype=np.int64)
    return roles


def safe_ball_xy(raw_observation: dict[str, Any]) -> np.ndarray:
    ball = np.asarray(raw_observation.get("ball", [0.0, 0.0, 0.0]), dtype=np.float32)
    if ball.size < 2:
        return np.zeros(2, dtype=np.float32)
    return ball[:2]


def safe_ball_direction(raw_observation: dict[str, Any]) -> np.ndarray:
    direction = np.asarray(raw_observation.get("ball_direction", [0.0, 0.0, 0.0]), dtype=np.float32)
    if direction.size < 2:
        return np.zeros(2, dtype=np.float32)
    return direction[:2]


def safe_owned_team(raw_observation: dict[str, Any]) -> int:
    try:
        return int(raw_observation.get("ball_owned_team", -1))
    except (TypeError, ValueError):
        return -1


def safe_owned_player(raw_observation: dict[str, Any]) -> int:
    try:
        return int(raw_observation.get("ball_owned_player", -1))
    except (TypeError, ValueError):
        return -1


def safe_game_mode(raw_observation: dict[str, Any]) -> int:
    try:
        return int(raw_observation.get("game_mode", 0))
    except (TypeError, ValueError):
        return 0


def safe_sticky_actions(agent_observation: dict[str, Any]) -> np.ndarray:
    sticky = np.asarray(agent_observation.get("sticky_actions", np.zeros(10)), dtype=np.float32)
    if sticky.ndim != 1:
        return np.zeros(10, dtype=np.float32)
    return sticky


def role_name(role_id: int) -> str:
    return ROLE_NAMES.get(int(role_id), f"role_{int(role_id)}")


def tactical_action_name(action_id: int) -> str:
    return TACTICAL_ACTION_NAMES.get(int(action_id), f"action_{int(action_id)}")


def role_one_hot(role_id: int) -> np.ndarray:
    vec = np.zeros(NUM_ROLES, dtype=np.float32)
    vec[int(np.clip(role_id, 0, NUM_ROLES - 1))] = 1.0
    return vec


def tactical_action_one_hot(action_id: int) -> np.ndarray:
    vec = np.zeros(NUM_TACTICAL_ACTIONS, dtype=np.float32)
    vec[int(np.clip(action_id, 0, NUM_TACTICAL_ACTIONS - 1))] = 1.0
    return vec


def clip_unit(value: float, limit: float = 1.0) -> float:
    return float(np.clip(value / max(limit, 1e-6), -1.0, 1.0))


def normalize_distance(value: float, scale: float = 2.0) -> float:
    return float(np.clip(value / max(scale, 1e-6), 0.0, 1.5))


def distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(a, dtype=np.float32) - np.asarray(b, dtype=np.float32)))


def nearest_controlled_slot_to_ball(raw_observation: dict[str, Any], role_map: ControlledRoleMap) -> int:
    if role_map.role_ids.size == 0:
        return -1
    left_team = safe_team_positions(raw_observation, "left_team")
    if left_team.shape[0] == 0:
        return -1
    controlled_positions = left_team[role_map.controlled_indices]
    ball_xy = safe_ball_xy(raw_observation)
    distances = np.linalg.norm(controlled_positions - ball_xy, axis=1)
    return int(np.argmin(distances)) if distances.size > 0 else -1


def nearest_opponent_distance_to_point(raw_observation: dict[str, Any], point_xy: np.ndarray) -> float:
    right_team = safe_team_positions(raw_observation, "right_team")
    if right_team.shape[0] == 0:
        return 1.0
    distances = np.linalg.norm(right_team - np.asarray(point_xy, dtype=np.float32), axis=1)
    return float(np.min(distances)) if distances.size > 0 else 1.0


def controlled_team_has_free_ball_advantage(raw_observation: dict[str, Any], role_map: ControlledRoleMap) -> bool:
    if role_map.role_ids.size == 0:
        return False
    obs = primary_observation(raw_observation)
    if obs is None or safe_owned_team(obs) != -1:
        return False

    left_team = safe_team_positions(obs, "left_team")
    right_team = safe_team_positions(obs, "right_team")
    if left_team.shape[0] == 0 or right_team.shape[0] == 0:
        return False

    ball_xy = safe_ball_xy(obs)
    ball_dir = safe_ball_direction(obs)
    controlled_positions = left_team[role_map.controlled_indices]
    controlled_dist = float(np.min(np.linalg.norm(controlled_positions - ball_xy, axis=1)))
    opponent_dist = float(np.min(np.linalg.norm(right_team - ball_xy, axis=1)))
    ball_speed = float(np.linalg.norm(ball_dir))
    return controlled_dist + 0.04 < opponent_dist and ball_speed > 0.01


def inferred_controlled_ball_owner_slot(raw_observation: dict[str, Any], role_map: ControlledRoleMap) -> int:
    if role_map.role_ids.size == 0:
        return -1

    owned_team = safe_owned_team(raw_observation)
    owned_player = safe_owned_player(raw_observation)
    if owned_team == 0:
        matches = np.where(role_map.controlled_indices == owned_player)[0]
        if matches.size > 0:
            return int(matches[0])

    nearest_slot = nearest_controlled_slot_to_ball(raw_observation, role_map)
    if nearest_slot < 0:
        return -1

    left_team = safe_team_positions(raw_observation, "left_team")
    ball_xy = safe_ball_xy(raw_observation)
    ball_dir = safe_ball_direction(raw_observation)
    nearest_player_xy = left_team[role_map.controlled_indices[nearest_slot]]
    ball_distance = distance(nearest_player_xy, ball_xy)
    ball_speed = float(np.linalg.norm(ball_dir))
    if ball_distance < 0.028 and ball_speed < 0.06:
        return nearest_slot
    return -1


def infer_controlled_indices(raw_observation: Any, team_size: int, requested_players: int) -> np.ndarray:
    items = observation_list(raw_observation)
    primary = items[0] if items else None
    selectable_indices = list(range(team_size))
    if primary is not None:
        roles = safe_roles(primary, team_size)
        selectable_indices = [idx for idx in selectable_indices if int(roles[idx]) != GOALKEEPER_ROLE]
        if not selectable_indices:
            selectable_indices = list(range(team_size))

    indices: list[int] = []
    for item in items:
        try:
            idx = int(item.get("active", -1))
        except (TypeError, ValueError):
            idx = -1
        if idx in selectable_indices and idx not in indices:
            indices.append(idx)

    target_count = min(len(selectable_indices), requested_players)
    if len(indices) < target_count:
        for idx in selectable_indices:
            if idx not in indices:
                indices.append(idx)
            if len(indices) >= target_count:
                break

    if indices:
        return np.asarray(indices[:target_count], dtype=np.int64)
    return np.asarray(selectable_indices[:target_count], dtype=np.int64)


def assign_tactical_roles(raw_observation: dict[str, Any], controlled_indices: np.ndarray) -> np.ndarray:
    if controlled_indices.size == 0:
        return np.zeros(0, dtype=np.int64)

    left_team = safe_team_positions(raw_observation, "left_team")
    roles = safe_roles(raw_observation, left_team.shape[0])
    role_ids = np.full(controlled_indices.shape[0], ROLE_CM, dtype=np.int64)
    remaining_slots = list(range(controlled_indices.shape[0]))

    gk_slots = [
        slot
        for slot, idx in enumerate(controlled_indices.tolist())
        if 0 <= idx < roles.size and int(roles[idx]) == GOALKEEPER_ROLE
    ]
    if gk_slots:
        gk_slot = gk_slots[0]
        role_ids[gk_slot] = ROLE_GK
        remaining_slots.remove(gk_slot)

    if not remaining_slots:
        return role_ids

    remaining_indices = controlled_indices[remaining_slots]
    remaining_positions = left_team[remaining_indices]

    st_slot = remaining_slots[int(np.argmax(remaining_positions[:, 0] - 0.08 * np.abs(remaining_positions[:, 1])))]
    role_ids[st_slot] = ROLE_ST
    remaining_slots.remove(st_slot)

    if len(remaining_slots) >= 3:
        remaining_indices = controlled_indices[remaining_slots]
        remaining_positions = left_team[remaining_indices]
        cm_slot = remaining_slots[int(np.argmin(np.abs(remaining_positions[:, 1]) + 0.4 * np.abs(remaining_positions[:, 0])))]
        role_ids[cm_slot] = ROLE_CM
        remaining_slots.remove(cm_slot)

    if len(remaining_slots) == 2:
        slot_a, slot_b = remaining_slots
        pos_a = left_team[controlled_indices[slot_a]]
        pos_b = left_team[controlled_indices[slot_b]]
        if pos_a[1] <= pos_b[1]:
            role_ids[slot_a] = ROLE_LD
            role_ids[slot_b] = ROLE_RD
        else:
            role_ids[slot_a] = ROLE_RD
            role_ids[slot_b] = ROLE_LD
    elif len(remaining_slots) == 1:
        slot = remaining_slots[0]
        pos = left_team[controlled_indices[slot]]
        role_ids[slot] = ROLE_LD if pos[1] <= 0.0 else ROLE_RD

    return role_ids


def build_controlled_role_map(raw_observation: Any, requested_players: int) -> ControlledRoleMap:
    obs = primary_observation(raw_observation)
    if obs is None:
        empty = np.zeros(0, dtype=np.int64)
        return ControlledRoleMap(empty, empty, tuple(), False)

    left_team = safe_team_positions(obs, "left_team")
    controlled_indices = infer_controlled_indices(raw_observation, left_team.shape[0], requested_players)
    role_ids = assign_tactical_roles(obs, controlled_indices)
    return ControlledRoleMap(
        controlled_indices=controlled_indices,
        role_ids=role_ids,
        role_names=tuple(role_name(r) for r in role_ids.tolist()),
        goalkeeper_controlled=bool(np.any(role_ids == ROLE_GK)),
    )


def role_home_position(role_id: int, ball_xy: np.ndarray, own_team_has_ball: bool) -> np.ndarray:
    ball_x = float(ball_xy[0])
    ball_y = float(ball_xy[1])
    if role_id == ROLE_GK:
        return np.asarray([-0.92, 0.0], dtype=np.float32)
    if role_id == ROLE_LD:
        return np.asarray([np.clip(-0.52 + 0.12 * ball_x, -0.7, -0.15), np.clip(-0.28 + 0.18 * ball_y, -0.55, -0.08)], dtype=np.float32)
    if role_id == ROLE_RD:
        return np.asarray([np.clip(-0.52 + 0.12 * ball_x, -0.7, -0.15), np.clip(0.28 + 0.18 * ball_y, 0.08, 0.55)], dtype=np.float32)
    if role_id == ROLE_CM:
        return np.asarray([np.clip(-0.12 + 0.18 * ball_x, -0.35, 0.22), np.clip(0.24 * ball_y, -0.26, 0.26)], dtype=np.float32)
    push = 0.18 if own_team_has_ball else 0.05
    return np.asarray([np.clip(0.18 + push + 0.22 * ball_x, -0.05, 0.7), np.clip(0.2 * ball_y, -0.22, 0.22)], dtype=np.float32)


def role_support_position(role_id: int, ball_xy: np.ndarray, own_team_has_ball: bool) -> np.ndarray:
    ball_x = float(ball_xy[0])
    ball_y = float(ball_xy[1])
    if role_id == ROLE_GK:
        return np.asarray([-0.86, np.clip(0.1 * ball_y, -0.12, 0.12)], dtype=np.float32)
    if role_id == ROLE_LD:
        offset_x = 0.02 if own_team_has_ball else -0.18
        return np.asarray([np.clip(ball_x + offset_x, -0.55, 0.24), np.clip(ball_y - 0.22, -0.62, -0.04)], dtype=np.float32)
    if role_id == ROLE_RD:
        offset_x = 0.02 if own_team_has_ball else -0.18
        return np.asarray([np.clip(ball_x + offset_x, -0.55, 0.24), np.clip(ball_y + 0.22, 0.04, 0.62)], dtype=np.float32)
    if role_id == ROLE_CM:
        offset_x = 0.10 if own_team_has_ball else -0.12
        return np.asarray([np.clip(ball_x + offset_x, -0.32, 0.42), np.clip(0.75 * ball_y, -0.24, 0.24)], dtype=np.float32)
    return np.asarray([np.clip(ball_x + (0.18 if own_team_has_ball else -0.02), 0.08, 0.88), np.clip(0.30 * ball_y, -0.24, 0.24)], dtype=np.float32)


def role_recover_position(role_id: int, ball_xy: np.ndarray) -> np.ndarray:
    ball_x = float(ball_xy[0])
    ball_y = float(ball_xy[1])
    if role_id == ROLE_GK:
        return np.asarray([-0.94, np.clip(0.2 * ball_y, -0.14, 0.14)], dtype=np.float32)
    if role_id == ROLE_LD:
        return np.asarray([np.clip(min(ball_x - 0.25, -0.28), -0.75, -0.2), np.clip(ball_y - 0.18, -0.6, -0.08)], dtype=np.float32)
    if role_id == ROLE_RD:
        return np.asarray([np.clip(min(ball_x - 0.25, -0.28), -0.75, -0.2), np.clip(ball_y + 0.18, 0.08, 0.6)], dtype=np.float32)
    if role_id == ROLE_CM:
        return np.asarray([np.clip(min(ball_x - 0.2, -0.18), -0.55, -0.05), np.clip(0.45 * ball_y, -0.24, 0.24)], dtype=np.float32)
    return np.asarray([np.clip(min(ball_x - 0.15, -0.02), -0.35, 0.15), np.clip(0.2 * ball_y, -0.2, 0.2)], dtype=np.float32)


def role_attack_position(role_id: int, ball_xy: np.ndarray) -> np.ndarray:
    ball_x = float(ball_xy[0])
    ball_y = float(ball_xy[1])
    if role_id == ROLE_GK:
        return np.asarray([-0.85, 0.0], dtype=np.float32)
    if role_id == ROLE_LD:
        return np.asarray([np.clip(ball_x + 0.02, -0.20, 0.42), np.clip(ball_y - 0.32, -0.68, -0.10)], dtype=np.float32)
    if role_id == ROLE_RD:
        return np.asarray([np.clip(ball_x + 0.02, -0.20, 0.42), np.clip(ball_y + 0.32, 0.10, 0.68)], dtype=np.float32)
    if role_id == ROLE_CM:
        return np.asarray([np.clip(ball_x + 0.12, -0.02, 0.52), np.clip(0.45 * ball_y, -0.32, 0.32)], dtype=np.float32)
    return np.asarray([np.clip(max(ball_x + 0.25, 0.30), 0.22, 0.90), np.clip(0.28 * ball_y, -0.26, 0.26)], dtype=np.float32)


def attacking_support_target(
    raw_observation: dict[str, Any],
    actual_player_index: int,
    role_id: int,
    *,
    aggressive: bool,
) -> np.ndarray:
    left_team = safe_team_positions(raw_observation, "left_team")
    right_team = safe_team_positions(raw_observation, "right_team")
    if left_team.shape[0] == 0 or actual_player_index >= left_team.shape[0]:
        return np.zeros(2, dtype=np.float32)

    ball_xy = safe_ball_xy(raw_observation)
    owned_team = safe_owned_team(raw_observation)
    owned_player = safe_owned_player(raw_observation)
    carrier_index = owned_player if owned_team == 0 and 0 <= owned_player < left_team.shape[0] else actual_player_index
    carrier_xy = left_team[carrier_index]
    player_xy = left_team[actual_player_index]

    if role_id == ROLE_ST:
        forward_offset = 0.26 if aggressive else 0.18
    elif role_id == ROLE_CM:
        forward_offset = 0.18 if aggressive else 0.12
    else:
        forward_offset = 0.12 if aggressive else 0.06

    side_anchor = float(player_xy[1])
    if abs(side_anchor) < 0.06:
        side_anchor = -0.18 if role_id == ROLE_LD else 0.18 if role_id == ROLE_RD else 0.0
    lane_y = float(np.clip(0.55 * ball_xy[1] + 0.45 * side_anchor, -0.44, 0.44))
    target = np.asarray(
        [
            np.clip(max(ball_xy[0] + forward_offset, carrier_xy[0] + 0.05), -0.25, 0.90),
            lane_y,
        ],
        dtype=np.float32,
    )

    teammate_mask = np.ones(left_team.shape[0], dtype=bool)
    teammate_mask[actual_player_index] = False
    if 0 <= carrier_index < teammate_mask.size:
        teammate_mask[carrier_index] = False
    other_teammates = left_team[teammate_mask]
    if other_teammates.shape[0] > 0:
        nearest_teammate_d = np.linalg.norm(other_teammates - target, axis=1)
        if float(np.min(nearest_teammate_d)) < 0.10:
            target[1] = float(np.clip(target[1] + (0.12 if target[1] <= 0.0 else -0.12), -0.48, 0.48))

    if right_team.shape[0] > 0:
        nearest_opp_idx = int(np.argmin(np.linalg.norm(right_team - target, axis=1)))
        nearest_opp = right_team[nearest_opp_idx]
        if distance(nearest_opp, target) < 0.12:
            target[0] = float(np.clip(target[0] + 0.06, -1.0, 1.0))
            target[1] = float(np.clip(target[1] + (0.10 if target[1] <= nearest_opp[1] else -0.10), -0.50, 0.50))

    return target.astype(np.float32)


def choose_relevant_target(role_id: int, tactical_action: int, ball_xy: np.ndarray, own_team_has_ball: bool) -> np.ndarray:
    if tactical_action == TACTICAL_CHASE_BALL:
        return np.asarray(ball_xy, dtype=np.float32)
    if tactical_action == TACTICAL_SUPPORT_BALL:
        return role_support_position(role_id, ball_xy, own_team_has_ball)
    if tactical_action == TACTICAL_RECOVER:
        return role_recover_position(role_id, ball_xy)
    if tactical_action == TACTICAL_ATTACK_SPACE:
        return role_attack_position(role_id, ball_xy)
    return role_home_position(role_id, ball_xy, own_team_has_ball)


def _controlled_outfield_order(raw_observation: dict[str, Any], role_map: ControlledRoleMap) -> tuple[int, int]:
    if role_map.role_ids.size == 0:
        return -1, -1
    left_team = safe_team_positions(raw_observation, "left_team")
    if left_team.shape[0] == 0:
        return -1, -1

    controlled_positions = left_team[role_map.controlled_indices]
    ball_xy = safe_ball_xy(raw_observation)
    outfield_mask = role_map.role_ids != ROLE_GK
    outfield_distances = np.where(outfield_mask, np.linalg.norm(controlled_positions - ball_xy, axis=1), 1e9)
    sorted_outfield_slots = np.argsort(outfield_distances)
    closest_slot = int(sorted_outfield_slots[0]) if sorted_outfield_slots.size > 0 else -1
    second_slot = int(sorted_outfield_slots[1]) if sorted_outfield_slots.size > 1 else -1
    return closest_slot, second_slot


def tactical_mode_indices(raw_observation: Any, role_map: ControlledRoleMap) -> np.ndarray:
    role_ids = role_map.role_ids
    if role_ids.size == 0:
        return np.zeros(0, dtype=np.int64)

    obs = primary_observation(raw_observation)
    if obs is None:
        default_modes = np.full(role_ids.shape[0], TACTICAL_MODE_SUPPORT_COVER, dtype=np.int64)
        default_modes[role_ids == ROLE_GK] = TACTICAL_MODE_GK
        return default_modes

    owned_team = safe_owned_team(obs)
    controlled_owner_slot = inferred_controlled_ball_owner_slot(obs, role_map)
    own_team_has_ball = owned_team == 0 or controlled_owner_slot >= 0
    free_ball_attack_phase = controlled_team_has_free_ball_advantage(obs, role_map)
    closest_slot, _ = _controlled_outfield_order(obs, role_map)

    mode_ids = np.full(role_ids.shape[0], TACTICAL_MODE_SUPPORT_COVER, dtype=np.int64)
    mode_ids[role_ids == ROLE_GK] = TACTICAL_MODE_GK

    for slot, actual_idx in enumerate(role_map.controlled_indices.tolist()):
        if role_ids[slot] == ROLE_GK:
            continue
        if own_team_has_ball:
            mode_ids[slot] = TACTICAL_MODE_ON_BALL if slot == controlled_owner_slot else TACTICAL_MODE_SUPPORT_ATTACK
        elif free_ball_attack_phase:
            mode_ids[slot] = TACTICAL_MODE_FIRST_DEFENDER if slot == closest_slot else TACTICAL_MODE_SUPPORT_ATTACK
        else:
            mode_ids[slot] = TACTICAL_MODE_FIRST_DEFENDER if slot == closest_slot else TACTICAL_MODE_SUPPORT_COVER
    return mode_ids


def tactical_action_mask(raw_observation: Any, role_map: ControlledRoleMap) -> np.ndarray:
    role_ids = role_map.role_ids
    mask = np.zeros((role_ids.size, NUM_TACTICAL_ACTIONS), dtype=np.float32)
    if role_ids.size == 0:
        return mask

    obs = primary_observation(raw_observation)
    if obs is None:
        mask[:, TACTICAL_HOLD] = 1.0
        return mask

    ball_xy = safe_ball_xy(obs)
    mode_ids = tactical_mode_indices(obs, role_map)
    right_team = safe_team_positions(obs, "right_team")
    controlled_team = safe_team_positions(obs, "left_team")

    for slot, (role_id, mode_id) in enumerate(zip(role_ids.tolist(), mode_ids.tolist())):
        player_mask = mask[slot]
        if role_id == ROLE_GK or mode_id == TACTICAL_MODE_GK:
            player_mask[[TACTICAL_HOLD, TACTICAL_PROGRESSIVE_PASS, TACTICAL_SAFE_RESET_PASS, TACTICAL_CLEAR_BALL]] = 1.0
            continue
        if mode_id == TACTICAL_MODE_ON_BALL:
            player_xy = controlled_team[role_map.controlled_indices[slot]] if controlled_team.shape[0] > role_map.controlled_indices[slot] else ball_xy
            nearest_opp_dist = (
                float(np.min(np.linalg.norm(right_team - player_xy, axis=1)))
                if right_team.shape[0] > 0
                else 1.0
            )
            allow_reset = float(ball_xy[0]) < -0.40 or nearest_opp_dist < 0.06
            player_mask[
                [
                    TACTICAL_HOLD,
                    TACTICAL_ATTACK_SPACE,
                    TACTICAL_PROGRESSIVE_PASS,
                    TACTICAL_CLEAR_BALL,
                ]
            ] = 1.0
            if allow_reset:
                player_mask[TACTICAL_SAFE_RESET_PASS] = 1.0
            if float(ball_xy[0]) > 0.18 and abs(float(ball_xy[1])) < 0.38:
                player_mask[TACTICAL_SHOOT_BALL] = 1.0
            continue
        if mode_id == TACTICAL_MODE_SUPPORT_ATTACK:
            player_mask[[TACTICAL_SUPPORT_BALL, TACTICAL_ATTACK_SPACE]] = 1.0
            if float(ball_xy[0]) < -0.45:
                player_mask[TACTICAL_HOLD] = 1.0
            continue
        if mode_id == TACTICAL_MODE_FIRST_DEFENDER:
            player_mask[[TACTICAL_HOLD, TACTICAL_CHASE_BALL, TACTICAL_RECOVER]] = 1.0
            continue
        player_mask[[TACTICAL_HOLD, TACTICAL_SUPPORT_BALL, TACTICAL_RECOVER]] = 1.0

    empty_rows = np.where(mask.sum(axis=1) <= 0.0)[0]
    if empty_rows.size > 0:
        mask[empty_rows, TACTICAL_HOLD] = 1.0
    return mask


def policy_head_indices(raw_observation: Any, role_map: ControlledRoleMap) -> np.ndarray:
    return tactical_mode_indices(raw_observation, role_map).astype(np.int64, copy=False)


def direction_action_from_delta(delta: np.ndarray, threshold: float = 0.025) -> int:
    dx = float(delta[0])
    dy = float(delta[1])
    if abs(dx) < threshold and abs(dy) < threshold:
        return ACTION_IDLE
    if abs(dx) < threshold:
        return ACTION_TOP if dy < 0.0 else ACTION_BOTTOM
    if abs(dy) < threshold:
        return ACTION_RIGHT if dx > 0.0 else ACTION_LEFT
    if dx > 0.0 and dy < 0.0:
        return ACTION_TOP_RIGHT
    if dx > 0.0 and dy > 0.0:
        return ACTION_BOTTOM_RIGHT
    if dx < 0.0 and dy < 0.0:
        return ACTION_TOP_LEFT
    return ACTION_BOTTOM_LEFT


def _best_forward_teammate(
    team: np.ndarray,
    player_index: int,
    raw_observation: dict[str, Any],
    *,
    min_forward_x: float = 0.02,
    allow_goalkeeper: bool = False,
    goalkeeper_index: int | None = None,
) -> tuple[int | None, float]:
    if team.shape[0] <= 1:
        return None, 0.0
    player_xy = team[player_index]
    best_idx = None
    best_score = -1e9
    for idx in range(team.shape[0]):
        if idx == player_index:
            continue
        if not allow_goalkeeper and goalkeeper_index is not None and idx == goalkeeper_index:
            continue
        teammate_xy = team[idx]
        if teammate_xy[0] < player_xy[0] + min_forward_x:
            continue
        receiver_space = nearest_opponent_distance_to_point(raw_observation, teammate_xy)
        link_length = distance(player_xy, teammate_xy)
        score = float(
            (teammate_xy[0] - player_xy[0])
            - 0.30 * abs(teammate_xy[1] - player_xy[1])
            + 0.35 * min(receiver_space, 0.4)
            - 0.12 * max(0.0, link_length - 0.32)
        )
        if score > best_score:
            best_score = score
            best_idx = idx
    return best_idx, best_score


def _best_reset_teammate(
    team: np.ndarray,
    player_index: int,
    *,
    goalkeeper_index: int | None = None,
    allow_goalkeeper: bool = True,
    danger_ball_x: float | None = None,
) -> tuple[int | None, float]:
    if team.shape[0] <= 1:
        return None, 0.0
    player_xy = team[player_index]
    best_idx = None
    best_score = -1e9
    for idx in range(team.shape[0]):
        if idx == player_index:
            continue
        if not allow_goalkeeper and goalkeeper_index is not None and idx == goalkeeper_index:
            continue
        teammate_xy = team[idx]
        backward_bonus = max(0.0, float(player_xy[0] - teammate_xy[0]))
        central_bonus = max(0.0, 0.4 - abs(float(teammate_xy[1])))
        spacing_bonus = max(0.0, 0.6 - distance(player_xy, teammate_xy))
        score = 0.55 * backward_bonus + 0.15 * central_bonus + 0.10 * spacing_bonus
        score -= 0.22 * max(0.0, 0.10 - teammate_xy[0])
        if goalkeeper_index is not None and idx == goalkeeper_index:
            score -= 0.30
            if danger_ball_x is not None and danger_ball_x > -0.35:
                score -= 0.30
        if score > best_score:
            best_score = score
            best_idx = idx
    return best_idx, best_score


def _carry_target(player_xy: np.ndarray, ball_xy: np.ndarray, nearest_opponent_xy: np.ndarray | None) -> np.ndarray:
    forward = np.asarray([0.20, 0.0], dtype=np.float32)
    if nearest_opponent_xy is None:
        return ball_xy + forward

    lateral_gap = float(ball_xy[1] - nearest_opponent_xy[1])
    lateral_push = 0.0
    if abs(lateral_gap) < 0.18:
        lateral_push = 0.08 if lateral_gap >= 0.0 else -0.08
    target = ball_xy + forward + np.asarray([0.0, lateral_push], dtype=np.float32)
    target[0] = float(np.clip(target[0], -1.0, 1.0))
    target[1] = float(np.clip(target[1], -0.34, 0.34))
    return target.astype(np.float32)


def tactical_to_low_level_action(
    tactical_action: int,
    role_id: int,
    actual_player_index: int,
    raw_observation: dict[str, Any],
    goalkeeper_controlled: bool = False,
) -> int:
    left_team = safe_team_positions(raw_observation, "left_team")
    right_team = safe_team_positions(raw_observation, "right_team")
    if left_team.shape[0] == 0 or actual_player_index >= left_team.shape[0]:
        return ACTION_IDLE

    player_xy = left_team[actual_player_index]
    ball_xy = safe_ball_xy(raw_observation)
    owned_team = safe_owned_team(raw_observation)
    own_team_has_ball = owned_team == 0
    opponent_has_ball = safe_owned_team(raw_observation) == 1
    ball_owned_player = safe_owned_player(raw_observation)
    self_has_ball = own_team_has_ball and ball_owned_player == actual_player_index
    if not self_has_ball and distance(player_xy, ball_xy) < 0.028 and float(np.linalg.norm(safe_ball_direction(raw_observation))) < 0.06:
        self_has_ball = True
        own_team_has_ball = True
    roles = safe_roles(raw_observation, left_team.shape[0])
    goalkeeper_candidates = np.flatnonzero(roles == GOALKEEPER_ROLE)
    goalkeeper_index = int(goalkeeper_candidates[0]) if goalkeeper_candidates.size > 0 else 0
    nearest_opponent_idx = int(np.argmin(np.linalg.norm(right_team - player_xy, axis=1))) if right_team.shape[0] > 0 else -1
    nearest_opponent_xy = right_team[nearest_opponent_idx] if nearest_opponent_idx >= 0 else None
    nearest_opponent_distance = distance(player_xy, nearest_opponent_xy) if nearest_opponent_xy is not None else 1.0
    target_xy = choose_relevant_target(role_id, tactical_action, ball_xy, own_team_has_ball)
    if not self_has_ball and own_team_has_ball and tactical_action in (TACTICAL_SUPPORT_BALL, TACTICAL_ATTACK_SPACE):
        target_xy = attacking_support_target(
            raw_observation,
            actual_player_index,
            role_id,
            aggressive=tactical_action == TACTICAL_ATTACK_SPACE,
        )

    in_danger_zone = float(ball_xy[0]) < -0.2 or nearest_opponent_distance < 0.08

    if self_has_ball and tactical_action == TACTICAL_HOLD:
        target_xy = _carry_target(player_xy, ball_xy, nearest_opponent_xy)

    if tactical_action == TACTICAL_PROGRESSIVE_PASS and self_has_ball:
        teammate_idx, teammate_score = _best_forward_teammate(
            left_team,
            actual_player_index,
            raw_observation,
            min_forward_x=0.04,
            allow_goalkeeper=False,
            goalkeeper_index=goalkeeper_index,
        )
        if teammate_idx is None:
            target_xy = _carry_target(player_xy, ball_xy, nearest_opponent_xy)
            action = direction_action_from_delta(target_xy - player_xy, threshold=0.015)
            return ACTION_RIGHT if action == ACTION_IDLE else action
        target_dist = distance(player_xy, left_team[teammate_idx])
        receiver_space = nearest_opponent_distance_to_point(raw_observation, left_team[teammate_idx])
        if teammate_score > 0.22 and target_dist > 0.42 and receiver_space > 0.18:
            return ACTION_LONG_PASS
        if target_dist > 0.25 and receiver_space > 0.12:
            return ACTION_HIGH_PASS
        return ACTION_SHORT_PASS

    if tactical_action == TACTICAL_SAFE_RESET_PASS and self_has_ball:
        allow_goalkeeper = goalkeeper_controlled and (
            role_id == ROLE_GK or float(ball_xy[0]) < -0.45 or (in_danger_zone and float(ball_xy[0]) < -0.2)
        )
        if not allow_goalkeeper and float(ball_xy[0]) > -0.18 and nearest_opponent_distance > 0.10:
            target_xy = _carry_target(player_xy, ball_xy, nearest_opponent_xy)
            action = direction_action_from_delta(target_xy - player_xy, threshold=0.015)
            return ACTION_RIGHT if action == ACTION_IDLE else action
        teammate_idx, _ = _best_reset_teammate(
            left_team,
            actual_player_index,
            goalkeeper_index=goalkeeper_index,
            allow_goalkeeper=allow_goalkeeper,
            danger_ball_x=float(ball_xy[0]),
        )
        if teammate_idx is None:
            return ACTION_LONG_PASS if in_danger_zone else ACTION_SHORT_PASS
        target_dist = distance(player_xy, left_team[teammate_idx])
        if teammate_idx == goalkeeper_index and not allow_goalkeeper:
            return ACTION_SHORT_PASS
        if in_danger_zone and target_dist > 0.35:
            return ACTION_HIGH_PASS
        return ACTION_SHORT_PASS

    if tactical_action == TACTICAL_SHOOT_BALL and self_has_ball:
        if float(ball_xy[0]) > 0.18 and abs(float(ball_xy[1])) < 0.34:
            return ACTION_SHOT
        target_xy = role_attack_position(role_id, ball_xy)

    if tactical_action == TACTICAL_CLEAR_BALL and self_has_ball:
        if role_id == ROLE_GK or float(ball_xy[0]) < -0.28 or (in_danger_zone and float(ball_xy[0]) < -0.1):
            return ACTION_LONG_PASS
        target_xy = _carry_target(player_xy, ball_xy, nearest_opponent_xy)
        action = direction_action_from_delta(target_xy - player_xy, threshold=0.015)
        return ACTION_RIGHT if action == ACTION_IDLE else action

    if self_has_ball:
        if nearest_opponent_distance < 0.06 and float(ball_xy[0]) < 0.0:
            return ACTION_LONG_PASS if role_id in (ROLE_GK, ROLE_LD, ROLE_RD) else ACTION_HIGH_PASS
        if float(ball_xy[0]) > 0.42 and abs(float(ball_xy[1])) < 0.30:
            return ACTION_SHOT
        move_delta = target_xy - player_xy
        action = direction_action_from_delta(move_delta)
        if action == ACTION_IDLE:
            carry_target = _carry_target(player_xy, ball_xy, nearest_opponent_xy)
            action = direction_action_from_delta(carry_target - player_xy, threshold=0.015)
        return ACTION_RIGHT if action == ACTION_IDLE else action

    if opponent_has_ball and distance(player_xy, ball_xy) < 0.035 and role_id != ROLE_GK:
        return ACTION_SLIDING

    move_delta = target_xy - player_xy
    action = direction_action_from_delta(move_delta)
    if action == ACTION_IDLE and tactical_action in (TACTICAL_CHASE_BALL, TACTICAL_SUPPORT_BALL, TACTICAL_RECOVER):
        move_delta = ball_xy - player_xy
        action = direction_action_from_delta(move_delta, threshold=0.015)
    return action


def translate_tactical_actions(
    tactical_actions: np.ndarray | list[int] | int,
    raw_observation: Any,
    role_map: ControlledRoleMap,
) -> list[int] | int:
    if role_map.role_ids.size == 0:
        if isinstance(tactical_actions, (list, tuple, np.ndarray)):
            flat = np.asarray(tactical_actions, dtype=np.int64).reshape(-1)
            return int(flat[0]) if flat.size else ACTION_IDLE
        return int(tactical_actions)

    actions = np.asarray(tactical_actions, dtype=np.int64).reshape(role_map.role_ids.size)
    obs_items = observation_list(raw_observation)
    primary = primary_observation(raw_observation)
    if primary is None:
        return actions.tolist()

    low_level_actions: list[int] = []
    for slot, (action_id, role_id, actual_idx) in enumerate(
        zip(actions.tolist(), role_map.role_ids.tolist(), role_map.controlled_indices.tolist())
    ):
        obs = obs_items[slot] if slot < len(obs_items) else primary
        low_level_actions.append(
            tactical_to_low_level_action(
                tactical_action=int(action_id),
                role_id=int(role_id),
                actual_player_index=int(actual_idx),
                raw_observation=obs if obs is not None else primary,
                goalkeeper_controlled=role_map.goalkeeper_controlled,
            )
        )

    if len(low_level_actions) == 1:
        return int(low_level_actions[0])
    return low_level_actions
