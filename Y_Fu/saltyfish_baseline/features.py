from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch


@dataclass(frozen=True)
class Simple115Slices:
    left_team: slice = slice(0, 22)
    left_dir: slice = slice(22, 44)
    right_team: slice = slice(44, 66)
    right_dir: slice = slice(66, 88)
    ball_pos: slice = slice(88, 91)
    ball_dir: slice = slice(91, 94)
    ball_owner: slice = slice(94, 97)
    active: slice = slice(97, 108)
    game_mode: slice = slice(108, 115)


SIMPLE115_SLICES = Simple115Slices()
SIMPLE115_DIM = 115


def _one_hot(index: int, size: int) -> np.ndarray:
    vector = np.zeros(size, dtype=np.float32)
    if 0 <= index < size:
        vector[index] = 1.0
    return vector


def _safe_player_index(raw_obs: dict[str, Any], key: str, team: np.ndarray) -> int:
    index = int(raw_obs.get(key, 0))
    return index if 0 <= index < len(team) else 0


def _nearest_relative(team: np.ndarray, reference: np.ndarray, *, exclude_index: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    if exclude_index is not None:
        mask = np.ones(len(team), dtype=bool)
        mask[exclude_index] = False
        candidates = team[mask]
    else:
        candidates = team
    if len(candidates) == 0:
        return np.zeros(2, dtype=np.float32), np.zeros(1, dtype=np.float32)
    deltas = candidates - reference
    distances = np.linalg.norm(deltas, axis=1)
    idx = int(np.argmin(distances))
    return deltas[idx].astype(np.float32), np.asarray([distances[idx]], dtype=np.float32)


def _compute_offside_flags(raw_obs: dict[str, Any]) -> np.ndarray:
    left_team = np.asarray(raw_obs["left_team"], dtype=np.float32)
    right_team = np.asarray(raw_obs["right_team"], dtype=np.float32)
    left_active = np.asarray(raw_obs.get("left_team_active", np.ones(len(left_team))), dtype=bool)
    right_active = np.asarray(raw_obs.get("right_team_active", np.ones(len(right_team))), dtype=bool)
    active_index = _safe_player_index(raw_obs, "active", left_team)

    opponent_x = right_team[right_active, 0]
    if opponent_x.size == 0:
        second_last_defender_x = 1.0
    elif opponent_x.size == 1:
        second_last_defender_x = float(opponent_x[0])
    else:
        second_last_defender_x = float(np.sort(opponent_x)[-2])

    ball_x = float(np.asarray(raw_obs["ball"], dtype=np.float32)[0])
    offside_line = max(ball_x, second_last_defender_x)
    flags = np.zeros(len(left_team), dtype=np.float32)
    for idx, player in enumerate(left_team):
        in_opponent_half = float(player[0]) > 0.0
        is_ahead = float(player[0]) > offside_line
        if idx == active_index:
            continue
        if bool(left_active[idx]) and in_opponent_half and is_ahead:
            flags[idx] = 1.0
    return flags


def build_engineered_features(raw_obs: dict[str, Any]) -> np.ndarray:
    left_team = np.asarray(raw_obs["left_team"], dtype=np.float32)
    left_dir = np.asarray(raw_obs["left_team_direction"], dtype=np.float32)
    right_team = np.asarray(raw_obs["right_team"], dtype=np.float32)
    left_tired = np.asarray(raw_obs.get("left_team_tired_factor", np.zeros(len(left_team))), dtype=np.float32)
    right_tired = np.asarray(raw_obs.get("right_team_tired_factor", np.zeros(len(right_team))), dtype=np.float32)
    left_yellow = np.asarray(raw_obs.get("left_team_yellow_card", np.zeros(len(left_team))), dtype=np.float32)
    right_yellow = np.asarray(raw_obs.get("right_team_yellow_card", np.zeros(len(right_team))), dtype=np.float32)
    left_roles = np.asarray(raw_obs.get("left_team_roles", np.zeros(len(left_team))), dtype=np.int64)
    sticky_actions = np.asarray(raw_obs.get("sticky_actions", np.zeros(10)), dtype=np.float32)

    active_index = _safe_player_index(raw_obs, "active", left_team)
    active_pos = left_team[active_index]
    active_dir = left_dir[active_index]
    active_tired = np.asarray([left_tired[active_index]], dtype=np.float32)
    active_yellow = np.asarray([left_yellow[active_index]], dtype=np.float32)
    active_role = _one_hot(int(left_roles[active_index]), 10)

    ball = np.asarray(raw_obs["ball"], dtype=np.float32)
    ball_dir = np.asarray(raw_obs["ball_direction"], dtype=np.float32)
    ball_relative = ball - np.asarray([active_pos[0], active_pos[1], 0.0], dtype=np.float32)

    own_goal_relative = np.asarray([-1.0 - active_pos[0], -active_pos[1]], dtype=np.float32)
    opp_goal_relative = np.asarray([1.0 - active_pos[0], -active_pos[1]], dtype=np.float32)
    nearest_teammate_relative, nearest_teammate_distance = _nearest_relative(
        left_team, active_pos, exclude_index=active_index
    )
    nearest_opponent_relative, nearest_opponent_distance = _nearest_relative(right_team, active_pos)
    team_centroid_relative = (left_team.mean(axis=0) - active_pos).astype(np.float32)
    opponent_centroid_relative = (right_team.mean(axis=0) - active_pos).astype(np.float32)

    ball_owned_team = int(raw_obs.get("ball_owned_team", -1))
    ball_owned_player = int(raw_obs.get("ball_owned_player", -1))
    ball_owner_one_hot = _one_hot(ball_owned_team + 1, 3)
    active_has_ball = np.asarray(
        [1.0 if ball_owned_team == 0 and ball_owned_player == active_index else 0.0],
        dtype=np.float32,
    )

    score = raw_obs.get("score", [0, 0])
    score_diff = np.asarray([np.clip((float(score[0]) - float(score[1])) / 5.0, -1.0, 1.0)], dtype=np.float32)
    steps_left = float(raw_obs.get("steps_left", 0)) / 3001.0

    features = np.concatenate(
        [
            active_pos.astype(np.float32),
            active_dir.astype(np.float32),
            active_tired,
            active_yellow,
            active_role,
            ball_relative.astype(np.float32),
            ball_dir.astype(np.float32),
            own_goal_relative,
            opp_goal_relative,
            nearest_teammate_relative,
            nearest_teammate_distance,
            nearest_opponent_relative,
            nearest_opponent_distance,
            team_centroid_relative,
            opponent_centroid_relative,
            _compute_offside_flags(raw_obs),
            sticky_actions.astype(np.float32),
            left_tired.astype(np.float32),
            right_tired.astype(np.float32),
            left_yellow.astype(np.float32),
            right_yellow.astype(np.float32),
            ball_owner_one_hot,
            active_has_ball,
            score_diff,
            np.asarray([steps_left], dtype=np.float32),
        ]
    )
    return features.astype(np.float32, copy=False)


def split_simple115(obs: torch.Tensor) -> dict[str, torch.Tensor]:
    return {
        "left_team": obs[:, SIMPLE115_SLICES.left_team],
        "left_dir": obs[:, SIMPLE115_SLICES.left_dir],
        "right_team": obs[:, SIMPLE115_SLICES.right_team],
        "right_dir": obs[:, SIMPLE115_SLICES.right_dir],
        "ball_state": obs[:, SIMPLE115_SLICES.ball_pos.start : SIMPLE115_SLICES.ball_owner.stop],
        "active": obs[:, SIMPLE115_SLICES.active],
        "game_mode": obs[:, SIMPLE115_SLICES.game_mode],
    }


ENGINEERED_FEATURE_DIM = int(
    build_engineered_features(
        {
            "left_team": np.zeros((11, 2), dtype=np.float32),
            "left_team_direction": np.zeros((11, 2), dtype=np.float32),
            "right_team": np.zeros((11, 2), dtype=np.float32),
            "ball": np.zeros(3, dtype=np.float32),
            "ball_direction": np.zeros(3, dtype=np.float32),
            "left_team_tired_factor": np.zeros(11, dtype=np.float32),
            "right_team_tired_factor": np.zeros(11, dtype=np.float32),
            "left_team_yellow_card": np.zeros(11, dtype=np.float32),
            "right_team_yellow_card": np.zeros(11, dtype=np.float32),
            "left_team_active": np.ones(11, dtype=np.float32),
            "right_team_active": np.ones(11, dtype=np.float32),
            "left_team_roles": np.zeros(11, dtype=np.int64),
            "sticky_actions": np.zeros(10, dtype=np.float32),
            "active": 0,
            "ball_owned_team": -1,
            "ball_owned_player": -1,
            "score": [0, 0],
            "steps_left": 3001,
        }
    ).shape[0]
)
