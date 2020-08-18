import numpy as np
from scipy import stats
import math
from numpy import linalg as LA          # linear algebra package from numpy
from collections import namedtuple

# ---------------------------------------------------------------------------------
#  Reward Functions
#
#
#  All rewards should take actions as their first positional argument.
#  All other arguments should be passed as keyword arguments.
#  All functions should except **kwargs as their last argument to avoid errors
#  when unknown arguments are passed.
#
# ---------------------------------------------------------------------------------
RewardComponent = namedtuple('RewardComponent', ['function', 'coefficient', 'kwargs'], defaults=[1, {}])

def l2_norm_reward(action, unfinished_charging_events, **kwargs):
  """ Returns a real reward values depending on the L2 norm of all active EVSE's actions """
  return (LA.norm(action, axis = 0)).sum()


def deadline_penalty(action, unfinished_charging_events, **kwargs):
  '''Returns job-incompleting penalty on current network state.'''
  penalty = 0
  # compute culmulative penalty term
  for remain, initial in unfinished_charging_events:
    penalty -= remain / initial
  return penalty