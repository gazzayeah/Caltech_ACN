import argparse
import gym
import gym_EV
import numpy as np
import itertools
import torch
import datetime
import matplotlib.pyplot as plt
from tensorboardX import SummaryWriter
from mpc import *
from network import Network
# start with small EVSE network
MAX_EV = 10

# maximum power assignment for individual EVSE
MAX_RATE = 6

# maximum power assignment for whole network
MAX_CAPACITY = 20

# oversubscription level from {1, 2, 3 ...}
MAX_INTENSITY = 3

# turning ratio of step-down transformer
TURN_RATIO = 4

# constraint type: 'SOC' or 'LINEAR'
CONSTRAINT_TYPE = 'SOC'

# set parser that defines invariant global variables
parser = argparse.ArgumentParser(description='PyTorch REINFORCE example')

# set ev env id to parser, we call ev env by this argument
parser.add_argument('--env-name', default="EV-v0",
                    help='name of the environment to run')

# set seed value to random generator, this is a tuning parameter for sepecific algorithms
parser.add_argument('--seed', type=int, default=456, metavar='N',
                    help='random seed (default: 456)')

# set maximum iteration for learning episodes (max 1000000 episodes by default)
parser.add_argument('--num_steps', type=int, default=1000000, metavar='N',
                    help='maximum number of steps (default: 1000000)')

parser.add_argument('--eval', type=bool, default=True,
                    help='Evaluates a policy a policy every 10 episode (default:True)')

# pack up parsers
args = parser.parse_args()

# Environment
# Removing Normalized Actions. 
# Another way to use it = actions * env.action_space.high[0] -> (https://github.com/sfujim/TD3). This does the same thing.
# (or add env._max_episode_steps to normalized_actions.py)
env = gym.make(args.env_name)
env.__init__(max_ev = MAX_EV, max_rate = MAX_RATE, max_capacity= MAX_CAPACITY, intensity = MAX_INTENSITY)

# Plant random seeds for customized initial state. This prevents wired thing from happening
torch.manual_seed(args.seed)
np.random.seed(args.seed)
env.seed(args.seed)

# Training Loop
total_numsteps = 0
# Initialize culmulative reward buffer
ereward_buffer = []

# local directory storing training and testing data (contains real_train and real_test folders)
dataDirectory = "../gym-EV_data" # change this PATH to your own.
log_folder_dir = 'runs/{}_EV={}_RATE={}_CAP={}_TYPE={}'.format(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"), MAX_EV, MAX_RATE, MAX_CAPACITY, CONSTRAINT_TYPE)
#TesnorboardX
writer = SummaryWriter(log_dir=log_folder_dir)

# construct ev network
N = Network(max_ev = MAX_EV, max_rate = MAX_RATE, max_capacity = MAX_CAPACITY, turning_ratio = TURN_RATIO, constraint_type = CONSTRAINT_TYPE)

# obtain constraint matrix and upper limite and phase information from EV network
infrastructure = N.constraint_matrix()

# initialize mpc algorithm
A = AdaptiveChargingOptimization(infrastructure, max_ev = MAX_EV, max_rate = MAX_RATE, max_capacity = MAX_CAPACITY, constraint_type = CONSTRAINT_TYPE)

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
        '''
        # get current state info: leading time and energy demanded
        stateDict = env.get_current_state
        maxRateVec = np.ones(MAX_EV) * MAX_RATE
        laxityPriority = np.argsort(stateDict['remain_time'] - stateDict['remain_time'] / maxRateVec)
         # Sample random action
        action = np.zeros(MAX_EV)
        for i in laxityPriority:  
            action[i] = MAX_RATE
            if np.sum(action) > MAX_CAPACITY:
                action[i] +=  MAX_CAPACITY - np.sum(action)
                break
        '''
        action = A.solve(env.get_active_state)[:,0]
        action = np.array([float((str(i))) for i in action])
        # Print random action
        #print("Episode  ({}): episode step {} taking action: {}".format(i_episode, episode_steps, action))
        
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
        #print("Next State: {} || Reward: {}".format(state, reward))        
    # append culmulative reward for current episode
    ereward_buffer.append(episode_reward)    
    # finish learning if the maximum steps of state evulution has been reached
    if total_numsteps > args.num_steps:
        break
    # Print episode iterating information
    print("Episode: (day) {}, total numsteps: {}, episode steps: {}, reward: {}".format(i_episode, total_numsteps, episode_steps, round(episode_reward, 2)))
    
    # Show performance
    if i_episode % 10 == 0 and args.eval == True:

        plt.figure('remain_power')
        remained_power = np.array(env.charging_result)
        initial_power = np.array(env.initial_bat)
        charged_power = initial_power - remained_power
        ub = len(remained_power) - 1
        lb = max(ub - 50, 0)
        ind = range(len(remained_power))
        p1 = plt.bar(ind[lb:ub], remained_power[lb:ub])
        p2 = plt.bar(ind[lb:ub], charged_power[lb:ub], bottom=remained_power[lb:ub])
        plt.legend((p1[0], p2[0]), ('Remained', 'Charged'))
        plt.savefig(log_folder_dir+'/episode='+str(i_episode)+'_remaining_power.png')

        # plot cumulative reward trend
        plt.figure('culmulative_reward')
        #print(env.reward_vec)
        p3 = plt.plot(range(1, len(env.reward_vec)+ 1), env.reward_vec)
        plt.savefig(log_folder_dir+'/cr_epi=' + str(i_episode) + '_ev=' + str(MAX_EV) + '.png')
        plt.close('all')        

        writer.add_scalar('reward/test', episode_reward, i_episode)

        print("----------------------------------------")
        print("Test Episode: {}, reward: {}".format(i_episode, round(episode_reward, 2)))
        print("----------------------------------------")    
    
    
env.close()

