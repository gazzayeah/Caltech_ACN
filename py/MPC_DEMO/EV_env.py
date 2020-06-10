import numpy as np
from scipy import stats
import math
from numpy import linalg as LA          # linear algebra package from numpy
import gym
gym.logger.set_level(40)                      # adjust Box precision to lower level (used in defining action and state space)
from gym import error, spaces, utils
from gym.utils import seeding
# RL packages
import random                                     # Handling random number generation
from random import choices
from collections import deque             # Ordered collection with ends


class EVEnv(gym.Env):
  '''
  Environment class that simulates EV charging mechanism based on the real data from ACN
  
  Args:
  max_ev (int): maximum EV slots (number of EVSEs) in the system;
  max_rate (float): maximum power assignment that one EVSE can deliver;
  max_capacity (float): maximum power transfer that whole system can tolerate.
  intensity (int): to control the level of oversubscription, merging n days of input data into one day
  '''

  def __init__(self, gamma = 13, phi = 1, max_ev=5, max_rate = 6, max_capacity=20, intensity = 1):
    # Parameter for reward function
    self.gamma = gamma
    self.phi = phi

    self.state = None
    self.n_EVs = max_ev
    self._max_episode_steps = 100000
    self.charging_reward = 0
    self.max_capacity = max_capacity
    self.max_rate = max_rate 
    self.intensity = intensity

    # store EV charging result when it's overdue
    self.charging_result = []
    # list that records initial battery stage == initial energy requested. It's for static data analysis.
    self.initial_bat = []
    # dictionary that registers initial battery stage == initial energy requested for current EV ast EVSE[key]. It's for dynamic storage.
    self.dic_bat = {}
    # store culmulative reward
    self.cul_reward = 0
    self.reward_vec = []
    self.numev = 0
    self.evperiod = 10

    # Specify the observation space
    lower_bound = np.array([0])
    # 24 is the upper bound of job time, up to 24 hours; 70 is energy upperbound in kWh
    upper_bound = np.array([24, 70])
    # repmat 0 to each element of state space: (|(d, e)| * n_Ev) = 2 * 5 = 10 in total (since they all have the same lower bound 0)
    low = np.tile(lower_bound, self.n_EVs * 2)
    # repmat (d_max, e_max) n_EV times to each element of state space
    high = np.tile(upper_bound, self.n_EVs)
    # observation matrix as the Box with lower and upper bound matrices defined in type float32.
    self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)
    
    # Specify the action space, same as above and maximum rate is 6 
    upper_bound = np.array([self.max_rate])
    low = np.tile(lower_bound, self.n_EVs)
    high = np.tile(upper_bound, self.n_EVs)
    self.action_space = spaces.Box(low=low, high=high, dtype=np.float32)

    # Reset time for new episode
    self.time = 0
    # Time interval is pre-defined from training and testing dataset. To change this, we need to recreat all training data from ACN Simulator.
    self.time_interval = 0.1
    # variable that stores data
    self.data = None

  def step(self, action):
    '''
    Core function that returns next state and reward given current state and action. Constraints of the system is considered for refining 
    original action.
    
    Args:
    action (np.array): vector of power assignment to each evses
    
    Returns:
    obs (np.array): a compact state space containing energy remaining and job remaining time for each EVSE job;
    reward (float): cumulative rewards from taking the input action;
    done (bool): flag to determine if an episode (the day) comes to an end or not;
    info (dict): empty, not used;
    refined_act (np.array) : action that has been forced to suffice the system constraints
    
    Raises: None
    '''
    # Update states according to a naive battery model
    # Time advances
    self.time = self.time + self.time_interval
    # Check if a new EV arrives
    for i in range(len(self.data)):
      # if there exist EVs whose arrival time is between current and last time step, then add these EVS
      if self.data[i, 0] > self.time - self.time_interval and self.data[i, 0] <= self.time:
        # Reject if all spots are full
        if np.where(self.state[:, 2] == 0)[0].size == 0:
          continue
        # Add a new active charging station
        else:
          # get the first empty EVSE index in the state matrix, regardless of phase type. 
          idx = np.random.choice(np.where(self.state[:, 2] == 0)[0], 1)[0]
          # Upload job time, SOC, battery stage and toggle activation entry
          self.state[idx, 0] = self.data[i, 1]
          self.state[idx, 1] = self.data[i, 2]
          self.state[idx, 2] = 1
          self.dic_bat[idx] = self.data[i, 2]

    # Allow non-zero actions only to those whose energy remaining is non-zero
    action[np.where(self.state[:, 2] == 0)[0]] = 0

    # Update remaining time and if negative, clip the value to 0
    time_result = self.state[:, 0] - self.time_interval
    self.state[:, 0] = time_result.clip(min=0)

    # Update battery
    charging_result = self.state[:, 1] - action * self.time_interval
    # Battery is full
    for item in range(len(charging_result)):
      if charging_result[item] < 0:
        action[item] = self.state[item, 1] / self.time_interval
    self.state[:, 1] = charging_result.clip(min=0)
    
    # initialize variable that record cumulative penalty from unfinished but overdue EVs
    penalty = 0
    
    # i is the index of whose activate toggle is 1: iterate overall all activating vehicles
    for i in np.nonzero(self.state[:, 2])[0]:
      # The EV has no remaining time: overdue
      if self.state[i, 0] == 0:
        self.numev += 1
        # store leaving EV's energy remaining
        self.charging_result = np.append(self.charging_result, self.state[i, 1])
        # store same EV's  initial battery (initial charging demand)
        self.initial_bat = np.append(self.initial_bat, self.dic_bat[i])
        
        # both overdue and unfinished
        if self.state[i, 1] > 0.0001:
          # penalty defined as proportion of uncharged energy against initial energy requested, times gamma
          penalty += self.gamma * self.state[i, 1] / self.dic_bat[i]
          print("Unfinished job detected at EVSE {}: {}".format(i, penalty))
        
        # Deactivate the EV and reset
        self.state[i, :] = 0
    
    # norm of action vector representing aggregated charging reward
    self.charging_reward = LA.norm(action[np.where(self.state[:, 2] == 1)[0]])

    # Update rewards: phi is the temperature coefficient for charging reward
    reward = (- penalty +  self.phi * self.charging_reward)
    self.cul_reward += reward
    '''
    if self.numev >= self.evperiod:
      self.reward_vec.append(self.cul_reward / self.evperiod)
      self.cul_reward = 0
      self.numev = 0
    '''
    # if timestep reaches the end of the day, the episode is finished
    done = True if self.time >= 24 else False
    # reshape state matrix to a row vector
    obs = self.state[:, 0:2].flatten()
    info = {}
    refined_act = action
    return obs, reward, done, info, refined_act
  
  @property 
  def get_current_state(self):
    """
    Obtain current state information partitioned into leading time and energy remaining
    
    Args:
    None
    
    Returns:
    (Dict[np.array, np.array]): array of current leading time, energy remaining and evse id
    
    Raises: 
    None
    """
    active = self.state[self.state[:,2] == 1]
    return {'remain_time': np.transpose(self.state[:,0:1])[0], 'remain_energy': np.transpose(self.state[:, 1:2])[0]}
  
  
  @property 
  def get_active_state(self):
    """
    Obtain active EV information partitioned into leading time and energy remaining and evse id
    
    Args:
    None
    
    Returns:
    (Dict[np.array, np.array]): array of current leading time and energy remaining
    
    Raises: 
    None
    """
    # append id of evse to state matrix
    idx = np.transpose([np.arange(self.n_EVs)])
    idxState = np.append(self.state, idx, axis = 1)
    active = idxState[self.state[:,2] == 1]
    return {'remain_time': np.transpose(active[:,0:1])[0], 'remain_energy': np.transpose(active[:, 1:2])[0], 'index': np.transpose(active[:, 3:4])[0]}  
  
  @property 
  def get_active_index(self):  
    """
    Obtain those EVSE ids whose job status are active.
    
    Args:
    None
    
    Returns:
    (np.array): array of active EV indices (EVSEs ids)
    
    Raises: 
    None
    """    
    # append id of evse to state matrix
    idx = np.transpose([np.arange(self.n_EVs)])
    idxState = np.append(self.state, idx, axis = 1)
    active = idxState[self.state[:,2] == 1]
    return np.transpose(active[:, 3:4])[0]

  def reset(self, dataDirectory, isTrain):
    '''
    Reset parameters of existing enviornment status to the initialized status. This includes
    initializing 1. self.time; 2. self.state; 3. self.data; 4. self.dic_bat.
    
    Args:
    isTrain (bool) : select training or testing dataset
    dataDirectory (string): directory storeing real time ACN data (training and testing)
    
    Returns:
    obs (np.array) : states without activation cloumn
    
    Raises: None
    '''    
    # Select a random day as the new episode and restart
    if isTrain:
      # only have 99 data files in real_train
      day = random.sample(range(0, 99), self.intensity)
      name = [dataDirectory + '/real_train/data' + str(d) + '.npy' for d in day]
    else:
       # only have 21 data files in real_train
      day = random.sample(range(0, 21), self.intensity)
      name = [dataDirectory + '/real_test/data' + str(d) + '.npy' for d in day]
    
    # initialize data with the first input file
    data = np.load(name[0])
    # Load the rest of data and initialize self.data
    for n in range(1, len(name)):
      data = np.load(name[n])
    self.data = data
    # find the index of the earliest arriving ev
    sidx = np.where(self.data[:, 0] == min(self.data[:, 0]))[0][0]
    # Initialize states and time
    self.state = np.zeros([self.n_EVs, 3])
    # Remaining time
    self.state[0, 0] = data[sidx, 1]
    # SOC
    self.state[0, 1] = data[sidx, 2]
    # The charging station is activated
    self.state[0, 2] = 1
    # append reward vec and reset culmulative reward
    self.reward_vec.append(self.cul_reward)
    self.cul_reward = 0
    # initialize timestep: equal to the first EV arrival time of the new episode
    self.time = data[sidx, 0]

    # append initial battery stage = initial energy demand
    self.dic_bat[0] = self.data[sidx, 2]
    # reshape state matrix to a row vector, activation column discarded in obs
    obs = self.state[:, 0:2].flatten()
    return obs
