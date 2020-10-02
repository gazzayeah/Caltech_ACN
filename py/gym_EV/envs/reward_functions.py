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

def l2_norm_reward(action, unfinished_charging_events, time, **kwargs):
  """ Returns a real reward values depending on the L2 norm of all active EVSE's actions """
  return (LA.norm(action, axis = 0)).sum()

def l1_norm_reward(action, unfinished_charging_events, time, **kwargs):
  """ Returns a real reward values depending on the L1 norm of all active EVSE's actions """
  return (np.sum(action, axis = 0)).sum()


def deadline_penalty(action, unfinished_charging_events, time, **kwargs):
  '''Returns job-incompleting penalty on current network state.'''
  penalty = 0
  # compute culmulative penalty term
  for remain, initial in unfinished_charging_events:
    penalty -= remain / initial
  return penalty


def quick_l1_norm_reward(action, unfinished_charging_events, time, **kwargs):
  """
  Objective function that encourages network to charge as fast as possible

  Args:
      Charging-rate matrix.

  Return:
     Output of objective function.
  """
  if "time" in kwargs.keys() and "period" in kwargs.keys():
      currentTime = kwargs["time"]
      period = kwargs["period"]
  else:
      raise ValueError("Current Time and Period Needed. \n")
  idx = int(round(currentTime / period))
  optimization_horizon = action.shape[1]
  weightVec = np.array([(240 - t) / 240 for t in range(240)])
  if len(action.shape) == 1:
    c = weightVec[idx]
    return c * action.sum()
  elif len(action.shape) == 2:
    c = weightVec[idx : idx + optimization_horizon]
    return c @ np.sum(action, axis = 0)
  else:
    raise ValueError("Erroneous action dimension {0}. Supposed to be 1 or 2.".format(len(action.shape)))
    
  
  
  