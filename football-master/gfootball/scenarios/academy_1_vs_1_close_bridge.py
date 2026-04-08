# coding=utf-8

from . import *


def build_scenario(builder):
  # A CPU-friendly defended bridge stage:
  # - one attacker vs one outfield defender
  # - no goalkeeper on the defending side
  # - ball starts already in an attacking position
  # This is meant to quickly teach "beat one defender, then finish".
  builder.config().game_duration = 120
  builder.config().deterministic = False
  builder.config().offsides = False
  builder.config().end_episode_on_score = True
  builder.config().end_episode_on_out_of_play = True
  builder.config().end_episode_on_possession_change = False
  builder.SetBallPosition(0.46, 0.0)

  builder.SetTeam(Team.e_Left)
  builder.AddPlayer(-1.0, 0.0, e_PlayerRole_GK)
  builder.AddPlayer(0.44, 0.0, e_PlayerRole_CB)

  builder.SetTeam(Team.e_Right)
  builder.AddPlayer(0.68, 0.0, e_PlayerRole_CB)
