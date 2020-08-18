import argparse
import gym
import gym_EV
import numpy as np
import itertools
import torch
import datetime
import matplotlib.pyplot as plt
from reward_functions import *
from hyper import *

# Environment
# Removing Normalized Actions. 
# Another way to use it = actions * env.action_space.high[0] -> (https://github.com/sfujim/TD3). This does the same thing.
# (or add env._max_episode_steps to normalized_actions.py)
env = gym.make(gymArgs.ENV_NAME)
env.__init__(dataArgs.START, 
             dataArgs.END_TRAIN, 
             reward = REWARD_FUNCTION, 
             max_ev = gymArgs.MAX_EV, 
             max_rate = gymArgs.MAX_RATE,
             intensity = gymArgs.INTENSITY, 
             phasePartition = netArgs.PHASE_PARTITION, 
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
        
        '''
        Learning Algorithms Should be Invoked Here
        
        Action above is made by random sampling from action space (for space detail see EV_env) ONLY FOR DEMOSTRATION.
        
        Action needs to be taken by training a policy from the algorithm.
        '''
        
        next_state, reward, done, info = env.step(action) # Step
        # record number of steps taken within the current episode
        episode_steps += 1
        # record number of steps taken in the whole learning process
        total_numsteps += 1
        # record cumulative reward from the learning process
        episode_reward += reward
        # Update state to next state
        state = next_state
        print(env.get_active_sessions(phaseType = -1))
        
        # Print current state information
        #print("Next State: {} || Reward: {} || New EVs : {}".format(state, reward, info))         
    # finish learning if the maximum steps of state evulution has been reached
    if i_episode > expArgs.TRAIN_EPISODES:
        break
    # Print episode iterating information
    print("Episode: (day) {}, total numsteps: {}, episode steps: {}, reward: {}".format(i_episode, total_numsteps, episode_steps, round(episode_reward, 2)))
    
  
    
env.close()

