import numpy as np
import collections
from scipy import stats
import math
import random                                     # Handling random number generation
from collections import deque             # Ordered collection with ends
from typing import List, Union
from gym_EV.envs.reward_functions import *
# gym packages
import gym
gym.logger.set_level(40)                      # adjust Box precision to lower level (used in defining action and state space)
from gym import error, spaces, utils
from gym.utils import seeding
import gym_EV.envs.acn_data_generation as DG
from MPC.network import *
# datetime modules
from datetime import timedelta
from datetime import datetime



class EVEnv(gym.Env):
  '''
  Environment class that simulates EV charging dynamics based on the data from ACN generator.
  
  Args:
  start (datetime object): start date of ACN data;
  end (datetime object): end date of ACN data;
  reward (List[RewardComponent]) : components of reward function, including reward functions, coefficient, kwargs;
  maxExternalCapacity (float): normalizing maximum of the external power time-series, eg. PV plants;
  intensity (int): to control the level of oversubscription, merging n days of input data into one day;
  phase_selection (Bool): follow real phase selection in ACN if True, or unirformly choosing phase if False;
  isRandomDate (Bool) : determine if ACN generator creates data with random date selection.
  '''

  def __init__(self, network = Network(),
               start : datetime = datetime(2018, 6, 15), 
               end : datetime = datetime(2018, 6, 22), 
               reward: List[RewardComponent] = [RewardComponent(l2_norm_reward, 1), RewardComponent(deadline_penalty, 0)], 
               maxExternalCapacity = 15,
               intensity : int = 1, 
               phase_selection = True, 
               isRandomDate = False):
    
    ########################################################
    #
    # (Static) Network Attributes
    #
    ########################################################    
    # initialize network settings
    self.network = network
    # determine number of EVSEs in the network
    self.n_EVs = network.max_ev
    # determines upper bound of action space (maximum power rate of an EVSE)
    self.maxRateVec = network.maxRateVec
    # determined the maximum external capacity
    self.maxExternalCapacity = maxExternalCapacity
    # endow each EVSE with a phase index
    self.phaseId = np.array([-1] * network.phase_partition[0] 
                            + [0] * network.phase_partition[1] 
                            + [1] * (self.n_EVs - sum(network.phase_partition)))    
    
    ########################################################
    #
    # (Dynamic) Network Attributes
    #
    ########################################################     
    # initialize vector recording arrving EV profile.It should be a 5-dimensional vector: arrival, duration, energy, max rate, capacity
    self.chargingSessions = np.array([])
    # store newly arriving EVs to the network
    self.arrivingEv = []
    # store EV charging result when it's overdue
    self.charging_result = [] 
    # record a real_time initial battery information of all active EVSEs in the network. It's for static data analysis.
    self.initial_state_dict = {}
    # Initialize states, for each EVSE, it controls: 1. activating toggle, 2.job remaining, 3. energy remaining 4. maximum rates intake.
    self.state = None  
    
    ########################################################
    #
    # (Static) Environment Attributes
    #
    ########################################################  
    # Components of reward function
    self.reward_configuration = reward
    # determine incoming EV arrivals
    self.intensity = intensity
    # start date of data extraction
    self.start = start
    # end date of data extraction
    self.end = end
    # decided if ACN generator selects phase randomly
    self.phaseSelection = phase_selection
    # decided if ACN generator selects date randomly
    self.isRandomDate = isRandomDate 
    # Time interval is pre-defined from training and testing dataset. To change this, we need to recreate all training data from ACN Simulator.
    self.time_interval = 0.1  
    # refine numerical accuracy in if statement 
    self.displacement = int(str(self.time_interval)[::-1].find('.'))
    # obtain network capacity time series
    self.solarCapacityDict = DG.get_solar_capacity(start, end, maxCapacity = maxExternalCapacity, period = self.time_interval)    
    
    ########################################################
    #
    # (Dynamic) Environment Attributes
    #
    ########################################################     
    # iteration of date from start to end
    self.tmpdate = start   
    # Reset time for new episode
    self.time = 0
    # initialize current timestamp
    self.dailyCapacityArray = np.zeros(int(round(24 / self.time_interval)))   
    # store culmulative reward
    self.cul_reward = 0
    # initialize vector that stores culmulative reward (eg. on daily basis)
    self.reward_vec = []
    # variable that stores data
    self.data = None     
    # initialize external capacity
    self.externalCapacity = 0
    # initialize exteranl capacity sampled by arrival
    self.externalCapacityByArrival = 0
    # initialize external rate vector
    self.externalRateVec = self.n_EVs * [0]
    # initialize exteranl infrastructure
    self.infrastructure = None
    # store EV's charging result that is unfinished
    self.unfinished_charging_result = []
    ## Specify the observation space
    lower_bound = np.array([0])
    ## 24 is the upper bound of job time, up to 24 hours; 70 is energy upperbound in kWh
    #upper_bound = np.array([24, 70])
    ## repmat 0 to each element of state space: (|(d, e)| * n_Ev) = 2 * 5 = 10 in total (since they all have the same lower bound 0)
    #low = np.tile(lower_bound, self.n_EVs * 2)
    ## repmat (d_max, e_max) n_EV times to each element of state space
    #high = np.tile(upper_bound, self.n_EVs)
    ## observation matrix as the Box with lower and upper bound matrices defined in type float32.
    #self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)
    
    # Specify the action space, same as above and maximum rate is 6 
    low = np.tile(lower_bound, self.n_EVs)
    high = self.maxRateVec
    self.action_space = spaces.Box(low=low, high=high, dtype=np.float32)


  
  @property 
  def get_current_state(self):
    '''return remaining time and energy of all EVSEs in the network'''
    active = self.state[self.state[:, 0] == 1]
    return {'remain_time': np.transpose(self.state[:,1 : 2])[0], 'remain_energy': np.transpose(self.state[:, 2 : 3])[0]}
  
  @property 
  def get_active_state(self):
    ''' return EVSE energy remaining, time remaining and index with non-zero energy, it may not include all active EVSEs'''
    return {'remain_time': self.state[np.where(self.state[:, 2] > 0.0001)[0], 1], 
            'remain_energy': self.state[np.where(self.state[:, 2] > 0.0001)[0], 2], 
            'index': np.where(self.state[:, 2] > 0.0001)[0]}  

  @property
  def get_active_number(self):
    ''' Get number of active EVSEs at current time step'''
    return np.where(self.state[:, 0] == 1)[0].size
  
  @property
  def get_arriving_ev(self):
    '''Get all EVs profile arriving at the network by two dimensional np.array'''
    return np.array(self.arrivingEv)
  
  @property
  def get_charging_sessions(self):
    '''get all past and present charging sessions'''
    return self.chargingSessions



  def get_current_active_sessions(self, phaseType = None, sampleType = "arrivalAdpt"):
    '''obtain next n active sessions by phase type, session includes [0] : index of EVSE connected to [1] : current time or arrival time 
       [2] : job duration of charging job; [3] : current energy remaining [4] maximum power rate intake [5] solar capacity value samples at EV's arrival time'''
    if phaseType == None:
      activeIdx = np.where((self.state[:, 0] == 1))[0] 
    elif phaseType in [-1, 0, 1]:
      activeIdx = np.where((self.state[:, 0] == 1) & (self.phaseId[:] == phaseType))[0] 
    else: 
      raise ValueError("Invalid phase type in get_current_active_sessions: should be in [ -1, 0, 1] but {0}".format(phaseType))
    # no active EV 
    if activeIdx.size == 0:
      return np.array([])
    else:      
      active = self.state[activeIdx, :]
      active = active[:, 1 : ].reshape(-1, active.shape[1] - 1)   
    if sampleType == "arrivalDelay":
      return np.concatenate((activeIdx.reshape(len(active), 1), np.zeros((len(active), 1)), active, self.externalCapacityByArrival * np.ones((len(active), 1))), axis = 1)
    elif sampleType in ["arrivalAdpt", "normal"]:
      return np.concatenate((activeIdx.reshape(len(active), 1), np.zeros((len(active), 1)), active, self.externalCapacity * np.ones((len(active), 1))), axis = 1)
    else:
      raise ValueError("Invalid Sample Type: {0}".format(sampleType))
  
  
  
  def get_nstep_charging_session(self, nStep = 0, phaseType = None, sampleType = "arrivalAdpt", perturbation = None):
    '''obtain next n active sessions (including current active sessions) by phase type, session includes [0] : index of EVSE connected to [1] : current time or arrival time 
        [2] : job duration of charging job; [3] : current energy remaining [4] maximum power rate intake [5] solar capacity value samples at EV's arrival time'''
    # sort data with ascending arrival sequence
    data = self.data  
    data[:, 1] = np.round(data[:, 1], self.displacement)
    ########################################################
    #
    # Extract part from data that has the same phaseType
    #
    ########################################################  
    if phaseType in [-1, 0, 1]:
      # convert phase ID list to set
      phaseSet = set(np.where(self.phaseId[:] == phaseType)[0])      
      # filter all data with phaseType, outpust follows : [arrival duration, energy] ~ phaseType
      data  = data[np.where((data[:, 0] == phaseType) & (data[:, 1] > self.time))[0]  , :]
      data = data[np.argsort(data[:, 1])]   
    else: 
      raise ValueError("Invalid phase type: should be in [-1, 0, 1] or None but {0}".format(phaseType))    
    # shift arrival time to the time delta with current time
    data[:, 1] -= self.time
    # only extract arrival time delta, duration, energy remaining, max rate and capacity
    data = data[:, 1 :]
    # obtain the first n sessions from data
    if len(data) >= nStep and nStep >= 0:
      data = data[0 : nStep, :]
    
    ## adding perturbation to simulate inaccurate EV-profile prediction
    #if type(perturbation) == tuple and len(perturbation) == 2:
      #noise = np.random.normal(perturbation[0], perturbation[1], data.shape)
      ## add noise to data
      #data += noise
      ## round the 1st and 2nd col of noise (arrival time and duration) to the 1 decimal place
      #data[:, 0 : 2] = np.round(data[:, 0 : 2], self.displacement)
      ##  clip arrival time to number no less than 0.1 (future)
      #data[:, 0] = np.clip(data[:, 0], a_min = 0.1)
      ## clip duration to the end of the day and 0
      #data[:, 1] = np.clip(data[:, 1], a_min = 0.1, a_max = np.round( (24 - self.time - data[:, 0]), self.displacement))
    #else:
      #raise ValueError("Invalid perturbation type, must be a 2-dim tuple but {0}".format(perturbation))    
    
    
    ########################################################
    #
    # Append the rectified sessions into active session list
    #
    ########################################################       
    # append all current active session
    activeSessions = self.get_current_active_sessions(phaseType = phaseType, sampleType = sampleType)
    for event in data:     
      # no active sessions registered
      if activeSessions.size == 0:
        # randomly select over all available EVSE for the event and insert to the 1st position of the event array
        idx = [int(round(i)) for i in phaseSet][0]
        event = np.append(np.array([[idx]]), np.array([event]))
        # add event to activeSessions
        activeSessions = np.array([event])       
      else:
        # find current EVSEs that have jobs registered
        occupiedEVSE = set(activeSessions[:, 0])
        # find current EVSEs that have no job registered
        availableEVSE = phaseSet - occupiedEVSE  
        # iterate over all registered EVSEs
        for idx in occupiedEVSE:
          # extract the maximum job deadline for all session in EVSE idx and compare it with the arrival time of the event
          if event[0] >= max(activeSessions[np.where(activeSessions[:, 0] == idx)[0], 1 : 3].sum(axis = 1)):
            # if the event arrives after a registered session
            availableEVSE.add(idx)     
        # there is no available EVSE for this session
        if availableEVSE == set():
          #raise ValueError("WARNING (from get_nstep_charging_session): new EV arrival is discarded due to deficient network feasibility. Discarded EV : {0}".format(event))
          print("WARNING (from get_nstep_charging_session): new EV arrival is discarded due to deficient network feasibility. Discarded EV : {0}".format(event))
          continue
        else:
          # randomly select over all available EVSE for the event and insert to the 1st position of the event array
          idx = [int(round(i)) for i in availableEVSE][0]
          event = np.append(np.array([[idx]]), np.array([event]))
          # add event to activeSessions
          activeSessions = np.append(activeSessions, np.array([event]), axis = 0)       
    if activeSessions.size == 0:
      return np.array([])
    else:
      # sort by EVSE index to have better visualization  
      return activeSessions[np.argsort(activeSessions[:, 1])]    
  
  
  
  def get_mpc_session_info(self, nStepList = [-1, -1, -1], sampleType = "normal"):
    '''Create charging session input for mpc module at current time, including current and next n sessions.
    sampleType includes [1] normal : sample external capacity for every minute [2] arrival : only sample external capacity
    when new EV arrives
    The output is a dictionary in the form of {charging session, rate constraint, capacity constraint}.
    Sample type includes normal : sampling solar capacity at every time interval; arrivalAdpt : only sampling at current time without knowing future values;
    arrivalDelay: only sampling at each new ev arrival'''
    if not np.all(np.array(nStepList) >= -1):
      raise ValueError("All elments in nStepList must be no less than -1, but have {0}".format(nStepList))
    mpcInfo = {}
    if type(nStepList) == int:
      if nStepList == 0:
        sessions = [self.get_nstep_charging_session(nStep = 0, phaseType = -1, sampleType = sampleType), 
                     self.get_nstep_charging_session(nStep = 0, phaseType = 0, sampleType = sampleType), 
                     self.get_nstep_charging_session(nStep = 0, phaseType = 1, sampleType = sampleType)]     
      elif nStepList < 0:
        sessions = [self.get_nstep_charging_session(nStep = -1, phaseType = -1, sampleType = sampleType), 
                     self.get_nstep_charging_session(nStep = -1, phaseType = 0, sampleType = sampleType), 
                     self.get_nstep_charging_session(nStep = -1, phaseType = 1, sampleType = sampleType)]             
      else:
        # sort data with ascending arrival sequence
        data = self.data
        data[:, 1] = np.round(data[:, 1], self.displacement)      
        # filter all data with phaseType, outpust follows : [arrival duration, energy] ~ phaseType
        data  = data[np.where(data[:, 1] > self.time)[0]  , :]     
        data = data[np.argsort(data[:, 1])]   
        # get the first n step EV sessions' phase information
        phaseInfo = data[0 : nStepList, 0].astype(int)
        # create number of EV arrivals at each phase
        sessions = [self.get_nstep_charging_session(nStep = len(np.where(phaseInfo == -1)[0]), phaseType = -1, sampleType = sampleType), 
                     self.get_nstep_charging_session(nStep = len(np.where(phaseInfo == 0)[0]), phaseType = 0, sampleType = sampleType), 
                     self.get_nstep_charging_session(nStep = len(np.where(phaseInfo == 1)[0]), phaseType = 1, sampleType = sampleType)]      
    else:
      sessions = [self.get_nstep_charging_session(nStep = nStepList[0], phaseType = -1, sampleType = sampleType), 
                   self.get_nstep_charging_session(nStep = nStepList[0], phaseType = 0, sampleType = sampleType), 
                   self.get_nstep_charging_session(nStep = nStepList[0], phaseType = 1, sampleType = sampleType)]
    # synthesize sessions into total session
    totalSessions = np.array([])
    for session in sessions:
      if session.size != 0:
        if totalSessions.size == 0:
          totalSessions = session
        else:
          totalSessions = np.append(totalSessions, session, axis = 0)
          
    if totalSessions.size == 0:
      mpcInfo["session_info"] = np.array([])
      mpcInfo["capacity_constraint"] = np.array([])  
      mpcInfo["rate_info"] = np.array([])  
    else:
      totalSessions = totalSessions[np.argsort(totalSessions[:, 1])]  
      mpcInfo["session_info"] = totalSessions
      dataLength = int(round(np.max(totalSessions[:, 1 : 3].sum(axis = 1)) / self.time_interval))
      # mate maxtrix should be num_ev * datalength
      mpcInfo["rate_info"] = np.zeros((self.n_EVs, dataLength))
      if sampleType == "normal":
        startIdx = int(round(self.time / self.time_interval))
        mpcInfo["capacity_constraint"] = self.dailyCapacityArray[startIdx : startIdx + dataLength]
        for session in totalSessions:
          # update rate info matrix
          mpcInfo["rate_info"][int(np.round(session[0])) , 
                               int(np.round(session[1] / self.time_interval)) : 
                               int(np.round((session[1] + session[2]) / self.time_interval))] = session[4]
      else:
        tmpCapacity = totalSessions[0][5]
        tmpArrivalTime = totalSessions[0][1]
        constraintSeries = np.array([])
        if tmpArrivalTime != 0:
          constraintSeries = np.append(constraintSeries, np.zeros(int(round(tmpArrivalTime / self.time_interval))))
        for session in totalSessions:
          # update rate info matrix
          mpcInfo["rate_info"][int(np.round(session[0])) , 
                               int(np.round(session[1] / self.time_interval)) : 
                               int(np.round((session[1] + session[2]) / self.time_interval))] = session[4]
          if session[1] != tmpArrivalTime:
            constraintSeries = np.append(constraintSeries, tmpCapacity * np.ones(int(round(((session[1] - tmpArrivalTime) / self.time_interval))))) 
            tmpCapacity = session[5]
            tmpArrivalTime = session[1]   
        constraintSeries = np.append(constraintSeries, totalSessions[-1][5] * np.ones(dataLength - len(constraintSeries)))
        mpcInfo["capacity_constraint"] = constraintSeries
    return mpcInfo



  def get_evse_id_by_phase(self, phaseType = None):
    '''obtain all EVSE id by specified phasetype'''
    if phaseType == None:
      return np.array([i for i in range(self.n_EVs)])
    elif phaseType in [-1, 0, 1]:
      return np.where(self.phaseId[:] == phaseType)[0] 
    else: 
      raise ValueError("Invalid phase type in get_evse_id_by_phase: should be in [-1, 0, 1] but {0}".format(phaseType))
    
    
    
    
  def get_charging_session_info(self, element = [1, 2, 3], phaseType = None):
    '''obtain all historical charging events by phase. Default elements include [1] : current time or arrival time 
        [2] : job duration of charging job; [3] : current energy remaining'''
    if self.chargingSessions.size == 0:
      return np.array([])
    if phaseType == None:
      return self.chargingSessions[:, element]
    elif phaseType in [-1, 0, 1]:
      return self.chargingSessions[np.where(self.chargingSessions[:, 0] == phaseType)[0] , element]
    else: 
      raise ValueError("Invalid phase type in get_previous_charging_events: should be in [-1, 0, 1] but {0}".format(phaseType))  
  
  

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
    # refine action by round robine to satisfy constraints
    #print(action)
    action = self.network.RoundRobin(action, externalCapacity = self.externalCapacity, externalRateVec = self.externalRateVec)
    #print(action)
    #print(self.externalCapacity)
    #print(self.externalRateVec)

    # Update remaining time and if negative, clip the value to 0
    time_result = self.state[:, 1] - self.time_interval
    self.state[:, 1] = time_result.clip(min = 0)

    # Update battery
    charging_result = self.state[:, 2] - action * self.time_interval
    # Clip charging result no less than 0
    self.state[:, 2] = charging_result.clip(min = 0)
    
    
    
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
    for idx in np.nonzero(self.state[:, 0])[0]:
      # The EV has no remaining time: overdue
      if self.state[idx, 1] < 0.0001:
        # append charging events
        self.charging_result.append((self.initial_state_dict[idx][3], self.state[idx, 2]))        
        # both overdue and unfinished
        if self.state[idx, 2] > 0.0001:
          # store leaving EV's energy remaining
          unfinished_charging_result.append((self.state[idx, 2], self.initial_state_dict [idx][3]))       
          self.unfinished_charging_result.append([self.state[idx, 2], self.initial_state_dict [idx][3]])    
          #print("Unfinished job detected at EVSE {}. state: {} || Initial Profile: {} ".format(idx, self.state[idx, 1 : 3], self.initial_state_dict [idx]))
        # clear the real time battery information dictionary at the index
        self.initial_state_dict [idx] = np.zeros(6)        
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
    self.time = np.round(self.time + self.time_interval, self.displacement)
    # info of arriving EVs
    info = []
    # Check if a new EV arrives
    for i in range(len(self.data)):
      # if there exist EVs whose arrival time is between current and last time step, then add these EVS
      if int(round(self.data[i, 1] / self.time_interval)) == int(round(self.time * 10)):
        # Reject if all spots are full
        if np.where((self.state[:, 0] == 0) & (self.phaseId[:]== self.data[i, 0]))[0].size == 0:
          print("WARNING (from gym step): new EV arrival is discarded due to deficient network feasibility. Discarded EV : {0}".format(self.data[i]))
          continue
        # Add a new active charging station
        else:
          # append arriving EV profile: [arrival, duration, energy, phase]
          info.append(self.data[i])
          # get the first empty EVSE index in the state matrix, regardless of phase type. 
          idx = np.where((self.state[:, 0] == 0) & (self.phaseId[:] == self.data[i, 0]))[0][0]
          # update (add) initial battery information of newly arriving EV to dicitonary
          self.initial_state_dict [idx] = self.data[i]          
          # The charging station is activated
          self.state[idx, 0] = 1    
          # Remaining time
          self.state[idx, 1] = self.data[i, 2]
          # State of charge
          self.state[idx, 2] = self.data[i, 3]
          # The maximum rate intake of current ev
          self.state[idx, 3] = self.data[i, 4]
          # Solar Capacity By Arrival
          self.externalCapacityByArrival = self.data[i, 5]
          # append this session in to charging session vector, discarding phase information
          if self.chargingSessions.size == 0:
            self.chargingSessions = np.array([self.data[i]])
          else:
            self.chargingSessions = np.append(self.chargingSessions, np.array([self.data[i]]), axis = 0)
    # append arriving EV profile vector
    self.arrivingEv += info 
    # obtain vector of maximum power intake of current time
    self.externalRateVec = self.state[:, 3]       
    # Update external capacity
    if self.time < 24:
      self.externalCapacity = self.dailyCapacityArray[int(round(self.time / self.time_interval))]
     
    
    

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
      charging_reward += component.coefficient * component.function(action, unfinished_charging_result, self.time, **component.kwargs)

    # Update rewards: phi is the temperature coefficient for charging reward
    self.cul_reward += charging_reward

    # if timestep reaches the end of the day, the episode is finished
    done = True if self.time >= 24 else False
    if done:
      # append reward vec and reset culmulative reward
      self.reward_vec.append(self.cul_reward)      
    # reshape state matrix to a row vector
    obs = self.state[:, 1 : 4].flatten()
    
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
    
    # initialize data matrix with activate prefix: [1] phase selection, [2] arrival time
    # [3] duration, [4] energy requested, [5] maximum power intake, [6] network capacity when ev arrives
    data = np.array([[0, 0, 0, 0, 0, 0]])
    currentDate = self.tmpdate
    # append data matrix based on date range
    for i in range(self.intensity):
      d = {1: np.array([])}
      # obtain valid daily data
      while d[1].size == 0:
        if self.isRandomDate:  
          # select a random date between start to end
          rndDate = random.randint(int(round(self.start.timestamp())), int(round(self.end.timestamp())))
          date = datetime(datetime.fromtimestamp(rndDate).year,datetime.fromtimestamp(rndDate).month, datetime.fromtimestamp(rndDate).day) 
        else:    
          date = self.tmpdate
          # make recursion of tmpdate between start date and end date
          if self.tmpdate + timedelta(days = 1) > self.end:
            self.tmpdate = self.start
          else:
            self.tmpdate = self.tmpdate + timedelta(days = 1)  
        # judge if today is weekdays
        if date.weekday() > 4:
          continue  
        else:
          #print("Date selected: {0}".format(date))
          d = DG.generate_events(date, 
                                 date + timedelta(days = 1), 
                                 solarCapacityDict = self.solarCapacityDict,
                                 phase_selection = self.phaseSelection, 
                                 period = self.time_interval)     
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
    data_idx_list = np.where(self.data[:, 1] == min(self.data[:, 1]))[0]
    # Initialize states, for each EVSE, it controls: 1. activating toggle, 2.job remaining, 3. energy remaining 4. maximum rates intake.
    self.state = np.zeros([self.n_EVs, 4])   
    # initialize initial battery information dictionary
    self.initial_state_dict  = {evse : np.zeros(6) for evse in range(self.n_EVs)}    
    for data_idx in data_idx_list:
      # get the first empty EVSE index in the state matrix, regardless of phase type. 
      sidx = np.where((self.state[:, 0] == 0) & (self.phaseId[:] == self.data[data_idx, 0]))[0][0]     
      # The charging station is activated
      self.state[sidx, 0] = 1    
      # Remaining time
      self.state[sidx, 1] = data[data_idx, 2]
      # State of charge
      self.state[sidx, 2] = data[data_idx, 3]
      # The maximum rate intake of current ev
      self.state[sidx, 3] = data[data_idx, 4]
      # Solar Capacity By Arrival
      self.externalCapacityByArrival = self.data[data_idx, 5]      
      # append initial battery stage = initial energy demand
      self.initial_state_dict [sidx] = self.data[data_idx]      
      # append this session in to charging session vector, discarding phase information
      if self.chargingSessions.size == 0:
        self.chargingSessions = np.array([self.data[data_idx]])
      else:
        self.chargingSessions = np.append(self.chargingSessions, np.array([self.data[data_idx]]), axis = 0)
    
    
    ########################################################
    #
    # Initialization of timestep, culmulative reward and battery information
    #
    ########################################################    
    
    self.cul_reward = 0
    # initialize timestep: equal to the first EV arrival time of the new episode
    self.time = np.round(min(self.data[:, 1]), self.displacement)
    # initialize external rate vector
    self.externalRateVec = self.state[:, 3]  
    #print(currentDate)
    # initialize external capacity
    #print(currentDate)
    #print(self.solarCapacityDict.keys())
    if currentDate in self.solarCapacityDict.keys():
      # initialize external capacity array at current date
      self.dailyCapacityArray = self.solarCapacityDict[currentDate] 
    else:    
      self.dailyCapacityArray = np.zeros(int(round(24 / self.time_interval)))    
    # initialize external capacity value at current time
    self.externalCapacity = self.dailyCapacityArray[int(round(self.time / self.time_interval))]   
    #print(self.externalCapacity)
    #print(self.externalCapacityByArrival)
    
    # reshape state matrix to a row vector, activation column discarded in obs
    obs = self.state[:, 1 : 4].flatten()
    return obs
