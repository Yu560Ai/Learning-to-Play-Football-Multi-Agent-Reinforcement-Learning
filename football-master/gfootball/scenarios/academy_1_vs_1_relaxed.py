# coding=utf-8

from . import *


def build_scenario(builder):
  # Shorter drill for faster feedback during curriculum transfer. The goal of
  # this stage is to learn carrying around one defender, not to maximize
  # episode length.
  builder.config().game_duration = 220
  builder.config().deterministic = False
  builder.config().offsides = False
  builder.config().end_episode_on_score = True
  builder.config().end_episode_on_out_of_play = True
  # Do not make every failed touch immediately terminal. This stage is meant
  # to teach carrying around a single defender.
  builder.config().end_episode_on_possession_change = False
  builder.SetBallPosition(0.02, 0.0)

  builder.SetTeam(Team.e_Left)
  builder.AddPlayer(-1.0, 0.0, e_PlayerRole_GK)
  builder.AddPlayer(0.0, 0.0, e_PlayerRole_CB)

  builder.SetTeam(Team.e_Right)
  # No goalkeeper in this bridge stage: the learning focus is beating a
  # single defender first, before adding goalkeeper pressure back in.
  builder.AddPlayer(0.22, 0.0, e_PlayerRole_CB)
