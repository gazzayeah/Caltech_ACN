import argparse
import gym
import gym_EV
import numpy as np
import itertools
import torch

# start with small EVSE network
MAX_EV = 5

# maximum power assignment for individual EVSE
MAX_RATE = 6

# maximum power assignment for whole network
MAX_CAPACITY = 10

# set parser that defines invariant global variables
parser = argparse.ArgumentParser(description='PyTorch REINFORCE example')

# set ev env id to parser, we call ev env by this argument
parser.add_argument('--env-name', default="EV-v0",
                    help='name of the environment to run')

# set seed value to random generator, this is a tuning parameter for sepecific algorithms
parser.add_argument('--seed', type=int, default=456, metavar='N',
                    help='random seed (default: 456)')

# set maximum iteration for learning episodes (max 1000000 episodes by default)
parser.add_argument('--num_steps', type=int, default=1, metavar='N',
                    help='maximum number of steps (default: 1000000)')

# pack up parsers
args = parser.parse_args()

# Environment
# Removing Normalized Actions. 
# Another way to use it = actions * env.action_space.high[0] -> (https://github.com/sfujim/TD3). This does the same thing.
# (or add env._max_episode_steps to normalized_actions.py)
env = gym.make(args.env_name)
env.__init__(max_ev = MAX_EV, max_rate = MAX_RATE, max_capacity= MAX_CAPACITY)

# Plant random seeds for customized initial state. This prevents wired thing from happening
torch.manual_seed(args.seed)
np.random.seed(args.seed)
env.seed(args.seed)

# Training Loop
total_numsteps = 0

# local directory storing training and testing data (contains real_train and real_test folders)
dataDirectory = "../gym-EV_data" # change this PATH to your own.

# count(1) returns 1, 2, 3 ...
for i_episode in itertools.count(1):
    episode_reward = 0
    episode_steps = 0
    done = False
    # reset to time (0-24) equal to the first EV errical time, not 0!
    state = env.reset(dataDirectory, isTrain=True)
    
    '''
    The while loop will break if and only if: 
    1. episode is finished (one day charging event) 
    or 2. iteration reaches num_steps.
    '''
    while not done:
        # Sample random action
        action = env.action_space.sample()  
        # Print random action
        print("Episode  ({}): episode step {} taking action: {}".format(i_episode, episode_steps, action))
        
        '''
        Learning Algorithms Should be Invoked Here
        
        Action above is made by random sampling from action space (for space detail see EV_env) ONLY FOR DEMOSTRATION.
        
        Action needs to be taken by training a policy from the algorithm.
        '''
        
        next_state, reward, done, _, refined_act = env.step(action) # Step
        # record number of steps taken within the current episode
        episode_steps += 1
        # record number of steps taken in the whole learning process
        total_numsteps += 1
        # record cumulative reward from the learning process
        episode_reward += reward
        # Update state to next state
        state = next_state
        
        # Print current state information
        print("Next State: {} || Reward: {}".format(state, reward))        
    
    # finish learning if the maximum steps of state evulution has been reached
    if total_numsteps > args.num_steps:
        break
    # Print episode iterating information
    print("Episode: (day) {}, total numsteps: {}, episode steps: {}, reward: {}".format(i_episode, total_numsteps, episode_steps, round(episode_reward, 2)))
    
env.close()

