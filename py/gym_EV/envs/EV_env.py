import numpy as np
from scipy import stats
import math
import random                                     # Handling random number generation
from random import choices
from collections import deque             # Ordered collection with ends
from typing import List, Union
from gym_EV.envs.reward_functions import *
# gym packages
import gym
gym.logger.set_level(40)                      # adjust Box precision to lower level (used in defining action and state space)
from gym import error, spaces, utils
from gym.utils import seeding
import gym_EV.envs.acn_data_generation as DG
# datetime modules
from datetime import timedelta
from datetime import datetime
import pytz
# Timezone of the ACN we are using.
timezone = pytz.timezone('America/Los_Angeles')



class EVEnv(gym.Env):
  '''
  Environment class that simulates EV charging dynamics based on the data from ACN generator.
  
  Args:
  start (datetime object): start date of ACN data;
  end (datetime object): end date of ACN data;
  reward (List[RewardComponent]) : components of reward function, including reward functions, coefficient, kwargs;
  max_ev (int): maximum EV slots (number of EVSEs) in the system;
  max_rate (float): maximum power assignment that one EVSE can deliver;
  intensity (int): to control the level of oversubscription, merging n days of input data into one day;
  phase_partition (List[int, int]): two-dimensioned vector, the first element of which determines number of EVSEs in AB line, the second of which determines that in BC line;
  phase_selection (Bool): follow real phase selection in ACN if True, or unirformly choosing phase if False;
  isRandomDate (Bool) : determine if ACN generator creates data with random date selection.
  '''

  def __init__(self, start : datetime = timezone.localize(datetime(2018, 5, 1)), 
               end : datetime = timezone.localize(datetime(2018, 6, 15)), 
               reward: List[RewardComponent] = [RewardComponent(l2_norm_reward, 1), RewardComponent(deadline_penalty, 0)], 
               max_ev : int = 5, 
               max_rate : int = 6,
               intensity : int = 1, 
               phasePartition : list = [2, 2], 
               phase_selection = True, 
               isRandomDate = False):
    
    # Components of reward function
    self.reward_configuration = reward
    # initialize EV network state
    self.state = None
    # determine number of EVSEs in the network
    self.n_EVs = max_ev
    # determines upper bound of action space (maximum power rate of an EVSE)
    self.max_rate = max_rate
    # determine incoming EV arrivals
    self.intensity = intensity
    # start date of data extraction
    self.start = start
    # end date of data extraction
    self.end = end
    # iteration of date from start to end
    self.tmpdate = start
    # endow each EVSE with a phase index
    self.phaseId = np.array([-1] * phasePartition[0]  + [0] * phasePartition[1] + [1] * (self.n_EVs - sum(phasePartition)))
    # decided if ACN generator selects phase randomly
    self.phaseSelection = phase_selection
    # decided if ACN generator selects date randomly
    self.isRandomDate = isRandomDate
    # initialize vector recoding arrving EV profile.It should be a 4-dimensional vector: arrival, duration, energy, phaseType
    self.chargingSessions = np.array([])
    
    # store newly arriving EVs to the network
    self.arrivingEv = []
    # store EV charging result when it's overdue
    self.charging_result = [] 
    # record a real_time initial battery information of all active EVSEs in the network. It's for static data analysis.
    self.initial_state_dict = {}
    # store culmulative reward
    self.cul_reward = 0
    # initialize vector that stores culmulative reward (eg. on daily basis)
    self.reward_vec = []
    
    # Reset time for new episode
    self.time = 0
    # Time interval is pre-defined from training and testing dataset. To change this, we need to recreat all training data from ACN Simulator.
    self.time_interval = 0.1
    # variable that stores data
    self.data = None    


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


  
  @property 
  def get_current_state(self):
    '''return remaining time and energy of all EVSEs in the network'''
    active = self.state[self.state[:,2] == 1]
    return {'remain_time': np.transpose(self.state[:,0:1])[0], 'remain_energy': np.transpose(self.state[:, 1:2])[0]}
  
  @property 
  def get_active_state(self):
    ''' return EVSE energy remaining, time remaining and index with non-zero energy, it may not include all active EVSEs'''
    return {'remain_time': self.state[np.where(self.state[:, 1] > 0.0001)[0], 0], 
            'remain_energy': self.state[np.where(self.state[:, 1] > 0.0001)[0], 1], 
            'index': np.where(self.state[:, 1] > 0.0001)[0]}  

  @property
  def get_active_number(self):
    ''' Get number of active EVSEs at current time step'''
    return np.where(self.state[:, 2] == 1)[0].size
  
  @property
  def get_arriving_ev(self):
    '''Get all EVs profile arriving at the network by two dimensional np.array'''
    return np.array(self.arrivingEv)
  
  @property
  def get_charging_sessions(self):
    '''get all past and present charging sessions'''
    return self.chargingSessions




  def get_current_active_sessions(self, phaseType = None):
    '''obtain current active sessions by phase type, session includes [1] : current time or arrival time 
        [2] : job duration of charging job; [3] : current energy remaining'''
    if phaseType == None:
      activeIdx = np.where(self.state[:, 2] == 1)[0]
      # no active EV 
      if activeIdx.size == 0:
        return np.array([])
      else:
        active = self.state[activeIdx, 0 : 2]
    elif phaseType in [-1, 0, 1]:
      activeIdx = np.where((self.state[:, 2] == 1) & (self.phaseId[:] == phaseType))[0] 
      # no active EV 
      if activeIdx.size == 0:
        return np.array([])
      else:      
        active = self.state[activeIdx, 0 : 2]
    else: 
      raise ValueError("Invalid phase type: should be in [None, -1, 0, 1] but {0}".format(phaseType))
    return np.concatenate((activeIdx.reshape(len(active), 1), np.zeros((len(active), 1)), active), axis = 1)
  

  def get_evse_id_by_phase(self, phaseType = None):
    '''obtain all EVSE id by specified phasetype'''
    if phaseType == None:
      return np.arange(0, len(self.phaseId))
    elif phaseType in [-1, 0, 1]:
      return np.where(self.phaseId[:] == phaseType)[0] 
    else: 
      raise ValueError("Invalid phase type: should be in [None, -1, 0, 1] but {0}".format(phaseType))
    
  def get_previous_charging_events(self, phaseType = None):
    '''obtain all historical charging events by phase. Events include [1] : current time or arrival time 
        [2] : job duration of charging job; [3] : current energy remaining'''
    if self.chargingSessions.size == 0:
      return np.array([])
    if phaseType == None:
      return self.chargingSessions[:, 0 : 3]
    elif phaseType in [-1, 0, 1]:
      return self.chargingSessions[np.where(self.chargingSessions[:, 3] == phaseType)[0] , 0 : 3]
    else: 
      raise ValueError("Invalid phase type: should be in [None, -1, 0, 1] but {0}".format(phaseType))  
  
  

  def step(self, action):
    '''
    State evolution that returns next state and reward given current state and action. Constraints of the system is not considered.
    
    Args:
    action (np.array): vector of power assignment to each evses
    
    Returns:
    obs (np.array): a compact state space containing energy remaining and job remaining time for each EVSE job;
    reward (float): cumulative rewards from taking the input action;
    done (bool): flag to determine if an episode (the day) comes to an end or not;
    info (list[np.array]): store all incoming EV profile between two consecutive time step;
    
    Raises: None
    '''
          
    ########################################################
    #
    # State evolution: time remaining, energy remaining 
    #
    ########################################################
          
    # Allow non-zero actions only to those whose energy remaining is non-zero
    action[np.where(self.state[:, 2] == 0)[0]] = 0

    # Update remaining time and if negative, clip the value to 0
    time_result = self.state[:, 0] - self.time_interval
    self.state[:, 0] = time_result.clip(min=0)

    # Update battery
    charging_result = self.state[:, 1] - action * self.time_interval
    # Clip charging result no less than 0
    self.state[:, 1] = charging_result.clip(min=0)
    
    
    
    ########################################################
    #
    # Checking all active EVSEs if a connected EV reaches job deadline.
    #
    # Record all nonzero energy remaining for each departing EV.
    #
    ########################################################
    
    # initialize vector storing unfinished charging events
    unfinished_charging_result = []
    # idx is the index of whose activate toggle is 1: iterate overall all activating vehicles
    for idx in np.nonzero(self.state[:, 2])[0]:
      # The EV has no remaining time: overdue
      if self.state[idx, 0] == 0:
        # append charging events
        self.charging_result.append((self.initial_state_dict [idx][2], self.state[idx, 1]))        
        # both overdue and unfinished
        if self.state[idx, 1] > 0.0001:
          # store leaving EV's energy remaining
          unfinished_charging_result.append((self.state[idx, 1], self.initial_state_dict [idx][2]))          
          print("Unfinished job detected at EVSE {}. state: {} || Initial Profile: {} ".format(idx, self.state[idx, 0 : 2], self.initial_state_dict [idx]))
        # clear the real time battery information dictionary at the index
        self.initial_state_dict [idx] = np.zeros(4)        
        # Deactivate the EV and reset
        self.state[idx, :] = 0        
    
    
    
    ########################################################
    #
    # New EV arrival: assign a random index by its phase type
    #
    # Store arriviing EV profile for data analysis (eg. LSTM)
    #
    ########################################################
    
    # Update states according to a naive battery model
    # Time advances
    self.time = self.time + self.time_interval    
    
    # info of arriving EVs
    info = []
    # Check if a new EV arrives
    for i in range(len(self.data)):
      # if there exist EVs whose arrival time is between current and last time step, then add these EVS
      if self.data[i, 0] > self.time - self.time_interval and self.data[i, 0] <= self.time:
        # Reject if all spots are full
        if np.where((self.state[:, 2] == 0) & (self.phaseId[:]== self.data[i, 3]))[0].size == 0:
          print("WARNING (from gym step): new EV arrival is discarded due to deficient network feasibility. Discarded EV : {0}".format(self.data[i]))
          continue
        # Add a new active charging station
        else:
          # append arriving EV profile: [arrival, duration, energy, phase]
          info.append(self.data[i])
          # get the first empty EVSE index in the state matrix, regardless of phase type. 
          idx = np.random.choice(np.where((self.state[:, 2] == 0) & (self.phaseId[:] == self.data[i, 3]))[0], 1)[0]
          # update (add) initial battery information of newly arriving EV to dicitonary
          self.initial_state_dict [idx] = self.data[i]          
          # Upload job time, SOC, battery stage and toggle activation entry
          self.state[idx, 0] = self.data[i, 1]
          self.state[idx, 1] = self.data[i, 2]
          self.state[idx, 2] = 1
          # append this session in to charging session vector, discarding phase information
          if self.chargingSessions.size == 0:
            self.chargingSessions = np.array([self.data[i, :]])
          else:
            self.chargingSessions = np.append(self.chargingSessions, np.array([self.data[i, :]]), axis = 0)
    # append arriving EV profile vector
    self.arrivingEv += info    
    
    

    ########################################################
    #
    # Compute instant reward from current state and actions.
    #
    # Check if one day is finished.
    #
    ########################################################    
    
    charging_reward = 0
    # norm of action vector representing aggregated charging reward
    for component in self.reward_configuration:
      charging_reward += component.coefficient * component.function(action, unfinished_charging_result, **component.kwargs)

    # Update rewards: phi is the temperature coefficient for charging reward
    self.cul_reward += charging_reward

    # if timestep reaches the end of the day, the episode is finished
    done = True if self.time >= 24 else False
    if done:
      # append reward vec and reset culmulative reward
      self.reward_vec.append(self.cul_reward)      
    # reshape state matrix to a row vector
    obs = self.state[:, 0:2].flatten()
    
    return obs, charging_reward, done, info



  def reset(self):
    '''
    Reset parameters of existing enviornment status to the initialized status. This includes
    initializing 1. self.time; 2. self.state; 3. self.data; 4. self.dic_bat.
    
    Args:
    None
    
    Returns:
    obs (np.array) : states without activation cloumn
    
    Raises: None
    '''
    ########################################################
    #
    # Create one-day data from ACN generator. Weekdays are considered.
    #
    ########################################################
    
    # initialize data matrix with activate prefix
    data = np.array([[0, 0, 0, 0]])

    # append data matrix based on date range
    for i in range(self.intensity):
      d = {1: np.array([])}
      # obtain valid daily data
      while d[1].size == 0:
        if self.isRandomDate:  
          # select a random date between start to end
          rndDate = random.randint(int(self.start.timestamp()), int(self.end.timestamp()))
          date = datetime(datetime.fromtimestamp(rndDate).year,datetime.fromtimestamp(rndDate).month, datetime.fromtimestamp(rndDate).day) 
        else:    
          date = self.tmpdate
          # make recursion of tmpdate between start date and end date
          if self.tmpdate + timedelta(days = 1) >= self.end:
            self.tmpdate = self.start
          else:
            self.tmpdate = self.tmpdate + timedelta(days = 1)  
        # judge if today is weekdays
        if date.weekday() > 4:
          continue  
        else:
          #print("Date selected: {0}".format(date))
          d = DG.generate_events(date, date + timedelta(days = 1), phase_selection = self.phaseSelection)     
      # append data matrix
      data = np.append(data, d[1], axis=0)
    # delete prefix
    data = np.delete(data, (0), axis=0)   
    # data format follows [arrival, duration, energy, phase]
    self.data = data
    
    
    
    ########################################################
    #
    # Append the first incomming EV profile to the network
    #
    ########################################################
    
    # find the index of the earliest arriving ev
    data_idx = np.where(self.data[:, 0] == min(self.data[:, 0]))[0][0]
    # Initialize states and time
    self.state = np.zeros([self.n_EVs, 3])
    
    # get the first empty EVSE index in the state matrix, regardless of phase type. 
    sidx = np.random.choice(np.where((self.state[:, 2] == 0) & (self.phaseId[:] == self.data[data_idx, 3]))[0], 1)[0]        
    
    # Remaining time
    self.state[sidx, 0] = data[data_idx, 1]
    # State of charge
    self.state[sidx, 1] = data[data_idx, 2]
    # The charging station is activated
    self.state[sidx, 2] = 1
    
    
    
    ########################################################
    #
    # Initialization of timestep, culmulative reward and battery information
    #
    ########################################################    
    
    self.cul_reward = 0
    # initialize timestep: equal to the first EV arrival time of the new episode
    self.time = data[data_idx, 0]
    
    # initialize initial battery information dictionary
    self.initial_state_dict  = {evse : np.zeros(4) for evse in range(self.n_EVs)}
    # append initial battery stage = initial energy demand
    self.initial_state_dict [sidx] = self.data[data_idx]
    # reshape state matrix to a row vector, activation column discarded in obs
    obs = self.state[:, 0:2].flatten()
    return obs