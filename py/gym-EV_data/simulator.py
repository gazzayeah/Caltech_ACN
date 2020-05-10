import numpy as np

class Simulator:
  """
  This is the class for EV simulator
  """

  def __init__(self):
    self.tau = 1 # hours between state updates

    self.max_steps = 24 / self.tau
    self.current_step = None

    self.state = None

    self.action_space




  def step(self, action):
    """
    This is the function to update the state and output the reward based on action
    :param action:
    :return:
    """
    reward = 0

    return reward


  def reset(self):


  def render(self, mode='human'):




  def close(self):