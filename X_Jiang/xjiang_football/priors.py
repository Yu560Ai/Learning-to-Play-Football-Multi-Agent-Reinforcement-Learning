from __future__ import annotations

import numpy as np

from xjiang_football.features import SIMPLE115_DIM, SIMPLE115_SLICES


def _direction_action(dx: float, dy: float) -> int:
    if abs(dx) < 1e-6 and abs(dy) < 1e-6:
        return 0
    angle = np.arctan2(-dy, dx)
    bins = [
        (0.3926990817, 5),
        (1.1780972451, 4),
        (1.9634954085, 3),
        (2.7488935719, 2),
        (3.1415926536, 1),
    ]
    if angle < -2.7488935719:
        return 1
    if angle < -1.9634954085:
        return 8
    if angle < -1.1780972451:
        return 7
    if angle < -0.3926990817:
        return 6
    for threshold, action in bins:
        if angle < threshold:
            return action
    return 1


def rule_based_single_player_action_from_obs(
    obs: np.ndarray,
    *,
    shoot_x_threshold: float = 0.18,
    shoot_y_threshold: float = 0.18,
    ball_control_tolerance: float = 0.08,
) -> int:
    base = np.asarray(obs, dtype=np.float32).reshape(-1)
    if base.size < SIMPLE115_DIM:
        return 0
    base = base[:SIMPLE115_DIM]

    left_team = base[SIMPLE115_SLICES.left_team].reshape(-1, 2)
    right_team = base[SIMPLE115_SLICES.right_team].reshape(-1, 2)
    ball_state = base[SIMPLE115_SLICES.ball_pos.start : SIMPLE115_SLICES.ball_owner.stop]
    active_one_hot = base[SIMPLE115_SLICES.active]
    active_index = int(np.argmax(active_one_hot)) if active_one_hot.size else 0
    active_index = int(np.clip(active_index, 0, left_team.shape[0] - 1))
    player = left_team[active_index]
    ball_xy = ball_state[:2] if ball_state.size >= 2 else np.zeros(2, dtype=np.float32)
    has_ball = float(np.linalg.norm(player - ball_xy)) <= ball_control_tolerance

    if has_ball:
        x = float(ball_xy[0])
        y = float(ball_xy[1])
        if x >= shoot_x_threshold and abs(y) <= shoot_y_threshold:
            return 9
        active_opponents = right_team[np.linalg.norm(right_team, axis=1) > 1e-6]
        if active_opponents.size:
            deltas = active_opponents - player
            ahead_mask = deltas[:, 0] > -0.02
            if np.any(ahead_mask):
                ahead = deltas[ahead_mask]
                distances = np.linalg.norm(ahead, axis=1)
                nearest = ahead[int(np.argmin(distances))]
                # If the defender is sitting in the central lane, demonstrate a
                # slight diagonal carry to create a better shooting angle.
                if nearest[0] < 0.28 and abs(nearest[1]) < 0.12:
                    side = -1.0 if nearest[1] >= 0.0 else 1.0
                    target = np.asarray([player[0] + 0.20, player[1] + 0.16 * side], dtype=np.float32)
                    return _direction_action(float(target[0] - player[0]), float(target[1] - player[1]))
        target = np.asarray([1.0, -0.15 * y], dtype=np.float32)
        return _direction_action(float(target[0] - player[0]), float(target[1] - player[1]))

    return _direction_action(float(ball_xy[0] - player[0]), float(ball_xy[1] - player[1]))


def batch_rule_based_single_player_actions(
    observations: np.ndarray,
    *,
    shoot_x_threshold: float = 0.18,
    shoot_y_threshold: float = 0.18,
    ball_control_tolerance: float = 0.08,
) -> np.ndarray:
    obs_array = np.asarray(observations, dtype=np.float32)
    return np.asarray(
        [
            rule_based_single_player_action_from_obs(
                obs,
                shoot_x_threshold=shoot_x_threshold,
                shoot_y_threshold=shoot_y_threshold,
                ball_control_tolerance=ball_control_tolerance,
            )
            for obs in obs_array
        ],
        dtype=np.int64,
    )
