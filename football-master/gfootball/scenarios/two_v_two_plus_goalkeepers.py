# coding=utf-8

from . import *


def build_scenario(builder):
  cfg = builder.config()
  cfg.game_duration = 400
  cfg.deterministic = False
  cfg.offsides = False
  cfg.end_episode_on_score = False
  cfg.end_episode_on_out_of_play = False
  cfg.end_episode_on_possession_change = False
  cfg.left_team_difficulty = 1.0
  cfg.right_team_difficulty = 1.0

  # Full-game-style custom scenario:
  # - left side: 2 controllable outfield players + built-in keeper
  # - right side: 2 built-in outfield players + built-in keeper
  # The ball starts near the left striker to encourage coordinated build-up
  # without collapsing the task into an academy drill.
  builder.SetBallPosition(-0.12, 0.0)

  builder.SetTeam(Team.e_Left)
  builder.AddPlayer(-1.0, 0.0, e_PlayerRole_GK, controllable=False)
  builder.AddPlayer(-0.16, -0.06, e_PlayerRole_CF)
  builder.AddPlayer(-0.32, 0.12, e_PlayerRole_RM)

  builder.SetTeam(Team.e_Right)
  builder.AddPlayer(-1.0, 0.0, e_PlayerRole_GK, controllable=False)
  builder.AddPlayer(-0.20, -0.10, e_PlayerRole_CB, controllable=False)
  builder.AddPlayer(-0.24, 0.10, e_PlayerRole_CM, controllable=False)
