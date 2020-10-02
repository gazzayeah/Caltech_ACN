import argparse
import gym
import MPC.mpc
import gym_EV
import numpy as np
import itertools
import torch
import datetime
import matplotlib.pyplot as plt
from hyper import *
from gym_EV.envs.reward_functions import *
# import MPC modules
from MPC.objective_functions import *
from MPC.network import *
from MPC.mpc import *


########################################################
#
# Construct charging network
#
######################################################## 
# construct ev network
NETWORK = Network(max_ev = gymArgs.MAX_EV, 
            maxRateVec = netArgs.MAX_RATE_VECTOR, 
            maxCapacity = netArgs.MAX_CAPACITY, 
            turning_ratio = netArgs.TURN_RATIO, 
            phase_partition = netArgs.PHASE_PARTITION,
            constraint_type = netArgs.CONSTRAINT_TYPE)

# Environment
# Removing Normalized Actions. 
# Another way to use it = actions * env.action_space.high[0] -> (https://github.com/sfujim/TD3). This does the same thing.
# (or add env._max_episode_steps to normalized_actions.py)
env = gym.make(gymArgs.ENV_NAME)
env.__init__(network = NETWORK, 
             start = dataArgs.START, 
             end = dataArgs.END_TRAIN, 
             reward = REWARD_FUNCTION, 
             maxExternalCapacity = netArgs.MAX_EXTERNAL_CAPACITY,
             intensity = gymArgs.INTENSITY, 
             phase_selection = dataArgs.PHASE_SELECT, 
             isRandomDate = gymArgs.RANDOM_DATE)

# Plant random seeds for customized initial state. This prevents wired thing from happening
torch.manual_seed(gymArgs.SEED)
np.random.seed(gymArgs.SEED)
env.seed(gymArgs.SEED)

# Training Loop
total_numsteps = 0

# count(1) returns 1, 2, 3 ...
for i_episode in itertools.count(1):
    episode_reward = 0
    episode_steps = 0
    done = False
    # reset to time (0-24) equal to the first EV errical time, not 0!
    state = env.reset()
    print(env.dailyCapacityArray)
    #print("Daily EV Sessions: {0}".format(env.data))
    #print("Daily EV Sessions: {0}".format(env.data[np.where((env.data[:, 0] == -1))[0] , :]))
    '''
    The while loop will break if and only if: 
    1. episode is finished (one day charging event) 
    or 2. iteration reaches num_steps.
    '''
    while not done:
        # Sample random action
        action = env.action_space.sample()  
        # Print random action
        #print("Episode  ({}): episode step {} taking action: {}".format(i_episode, episode_steps, action))
        #print(env.externalCapacity)
        #print(env.externalCapacity == env.dailyCapacityArray[int( env.time/ 0.1)])
        '''
        Learning Algorithms Should be Invoked Here
        
        Action above is made by random sampling from action space (for space detail see EV_env) ONLY FOR DEMOSTRATION.
        
        Action needs to be taken by training a policy from the algorithm.
        '''
        
        next_state, reward, done, info = env.step(action) # Step
        #print("New ev arrival information: {0}".format(info))
        # record number of steps taken within the current episode
        episode_steps += 1
        # record number of steps taken in the whole learning process
        total_numsteps += 1
        # record cumulative reward from the learning process
        episode_reward += reward
        # Update state to next state
        state = next_state
        #print("Current Active Sessions: \n {0}".format(env.get_current_active_sessions(phaseType = None)))
        #nstep = -1
        #print("Next {0}-step Active Sessions: \n\r {1}".format(nstep, env.get_nstep_charging_session(nStep = nstep, phaseType = None)))
        
        # Print current state information
        #print("Next State: {} || Reward: {} || New EVs : {}".format(state, reward, info))         
    # Print episode iterating information
    print("Episode: (day) {}, total numsteps: {}, episode steps: {}, reward: {}".format(i_episode, total_numsteps, episode_steps, round(episode_reward, 2)))
    # finish learning if the maximum steps of state evulution has been reached
    if i_episode >= expArgs.TRAIN_EPISODES:
        break    
    
  
    
env.close()

