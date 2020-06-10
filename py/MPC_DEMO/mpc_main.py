import argparse
import datetime
import itertools

import gym
import gym_EV
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from tensorboardX import SummaryWriter

from benchmark import *
from mpc import *
from network import Network

# start with small EVSE network
MAX_EV = 12  # action dimension

# maximum power assignment for individual EVSE
MAX_RATE = 6

# maximum power assignment for whole network
# less than MAX_EV*MAX_RATE
MAX_CAPACITY = 40

# oversubscription level from {1, 2, 3 ...}
MAX_INTENSITY = 3

# turning ratio of step-down transformer
TURN_RATIO = 4

# constraint type: 'SOC'(three phase) or 'LINEAR'(single phase)
CONSTRAINT_TYPE = 'SOC'

# toggle between EDF, LLF and MPC algorithms
ACTION = 'MPC'

# set parser that defines invariant global variables
parser = argparse.ArgumentParser(description='PyTorch REINFORCE example')

# set ev env id to parser, we call ev env by this argument
parser.add_argument('--env-name', default="EV-v0",
                    help='name of the environment to run')

# set seed value to random generator, this is a tuning parameter for sepecific algorithms
parser.add_argument('--seed', type=int, default=456, metavar='N',
                    help='random seed (default: 456)')

# temperature coefficient gamma controls weights of penalty of failing to complete ev job
parser.add_argument('--gamma', type=float, default = 13, metavar='N',
                    help='control temperature of incomplete job penalty (default: 13)')

#  temperature coefficient phi controls weights of reward of charging action
parser.add_argument('--phi', type=float, default = 1, metavar='N',
                    help='control temperature of charging reward (default: 1)')

# set maximum iteration for learning episodes (max 1000000 episodes by default)
parser.add_argument('--num_steps', type=int, default=1000000, metavar='N',
                    help='maximum number of steps (default: 1000000)')

# if true, process evaluation for every 10 episode
parser.add_argument('--eval', type=bool, default=True,
                    help='Evaluates a policy a policy every 10 episode (default:True)')

# pack up parsers
args = parser.parse_args()

# Environment
# Removing Normalized Actions. 
# Another way to use it = actions * env.action_space.high[0] -> (https://github.com/sfujim/TD3). This does the same thing.
# (or add env._max_episode_steps to normalized_actions.py)
env = gym.make(args.env_name)
env.__init__(args.gamma, args.phi, max_ev = MAX_EV, max_rate = MAX_RATE, max_capacity= MAX_CAPACITY, intensity = MAX_INTENSITY)


# Plant random seeds for customized initial state. This prevents wired thing from happening
torch.manual_seed(args.seed)
np.random.seed(args.seed)
env.seed(args.seed)

# Training Loop
total_numsteps = 0

# local directory storing training and testing data (contains real_train and real_test folders)
dataDirectory = "../gym-EV_data"  # change this PATH to your own.
log_folder_dir = 'runs/{}_AL={}_EV={}_RATE={}_CAP={}_TYPE={}'.format(
    datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"), ACTION, MAX_EV, MAX_RATE, MAX_CAPACITY, CONSTRAINT_TYPE)
dataFilePath = 'runs/data/'

#TesnorboardX
writer = SummaryWriter(log_dir=log_folder_dir)

# construct ev network
N = Network(max_ev=MAX_EV, max_rate=MAX_RATE, max_capacity=MAX_CAPACITY, turning_ratio=TURN_RATIO,
            constraint_type=CONSTRAINT_TYPE)

# obtain constraint matrix and upper limite and phase information from EV network
infrastructure = N.infrastructure_info()

# initialize mpc algorithm
A = AdaptiveChargingOptimization(infrastructure, [ObjectiveComponent(quick_charge)], max_ev=MAX_EV, max_rate=MAX_RATE, max_capacity=MAX_CAPACITY,
                                 constraint_type=CONSTRAINT_TYPE)

# count(1) returns 1, 2, 3 ...
# training
test_creward_vec = []
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
    train_loss = 0
    while not done:
        # initialize action vector
        action = np.zeros(MAX_EV)
        # select algorithms that computes action vector
        if ACTION == 'LLF':
            action = LLFEDF(env.get_current_state, infrastructure, MAX_EV, MAX_RATE, MAX_CAPACITY, CONSTRAINT_TYPE,  algorithm = 'LLF')
        elif ACTION == 'EDF':
            action = LLFEDF(env.get_current_state, infrastructure, MAX_EV, MAX_RATE, MAX_CAPACITY, CONSTRAINT_TYPE,  algorithm = 'EDF')
        else:
            # obtain optimal action for current time step from MPC
            action = A.solve(env.get_active_state)[:,0]
            action = np.array([float((str(i))) for i in action])# covert to float ndarray

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
    # finish learning if the maximum steps of state evulution has been reached
    # run 100 days
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
        p3 = plt.plot(range(len(env.reward_vec) - 1), env.reward_vec[1:])
        plt.savefig(log_folder_dir+'/cr_epi=' + str(i_episode) + '_ev=' + str(MAX_EV) + '.png')
        plt.close('all')        

        writer.add_scalar('reward/test', episode_reward, i_episode)
    
    
    if total_numsteps > args.num_steps:
        trainname = dataFilePath + "train_creward_{}.npy".format(ACTION)
        with open(trainname, 'wb') as f:
            np.save(f, env.reward_vec[1:])
        testname = dataFilePath + "test_creward_{}.npy".format(ACTION)
        with open(testname, 'wb') as f:
            np.save(f, test_creward_vec)        
        break

env.close()












