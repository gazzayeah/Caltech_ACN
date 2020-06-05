import argparse
import datetime
import itertools

import gym
import gym_EV
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

from MPC.agent_model import LearningAgent
from MPC.benchmark import *
from MPC.mpc import *
from MPC.network import Network



testing_num_days = 1000
plot_loss = True
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
ACTION = 'LLF'

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
env1 = gym.make(args.env_name)
env1.__init__(args.gamma, args.phi, max_ev = MAX_EV, max_rate = MAX_RATE, max_capacity= MAX_CAPACITY, intensity = MAX_INTENSITY)
env1.seed(args.seed)

env2 = gym.make(args.env_name)
env2.__init__(args.gamma, args.phi, max_ev = MAX_EV, max_rate = MAX_RATE, max_capacity= MAX_CAPACITY, intensity = MAX_INTENSITY)
env2.seed(args.seed)

env3 = gym.make(args.env_name)
env3.__init__(args.gamma, args.phi, max_ev = MAX_EV, max_rate = MAX_RATE, max_capacity= MAX_CAPACITY, intensity = MAX_INTENSITY)
env3.seed(args.seed)

# Plant random seeds for customized initial state. This prevents wired thing from happening
torch.manual_seed(args.seed)
np.random.seed(args.seed)


# Training Loop
total_numsteps = 0

# local directory storing training and testing data (contains real_train and real_test folders)
dataDirectory = "../gym-EV_data"  # change this PATH to your own.
log_folder_dir = 'runs/{}_AL={}_EV={}_RATE={}_CAP={}_TYPE={}'.format(
    datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"), ACTION, MAX_EV, MAX_RATE, MAX_CAPACITY, CONSTRAINT_TYPE)
dataFilePath = 'runs/data/'

#TesnorboardX
#writer = SummaryWriter(log_dir=log_folder_dir)

# construct ev network
N = Network(max_ev=MAX_EV, max_rate=MAX_RATE, max_capacity=MAX_CAPACITY, turning_ratio=TURN_RATIO,
            constraint_type=CONSTRAINT_TYPE)

# obtain constraint matrix and upper limite and phase information from EV network
infrastructure = N.infrastructure_info()

# initialize mpc algorithm
A = AdaptiveChargingOptimization(infrastructure, max_ev=MAX_EV, max_rate=MAX_RATE, max_capacity=MAX_CAPACITY,
                                 constraint_type=CONSTRAINT_TYPE)

# initialize the agent models
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
agent1 = LearningAgent(action_dim=MAX_EV, state_dim=MAX_EV * 2, hidden_dim=MAX_EV).to(device)
agent1.load_state_dict(torch.load('../image/agent_{}.pt'.format('LLF'), map_location=torch.device('cpu')))

agent2 = LearningAgent(action_dim=MAX_EV, state_dim=MAX_EV * 2, hidden_dim=MAX_EV).to(device)
agent2.load_state_dict(torch.load('../image/agent_{}.pt'.format('EDF'), map_location=torch.device('cpu')))

agent3 = LearningAgent(action_dim=MAX_EV, state_dim=MAX_EV * 2, hidden_dim=MAX_EV).to(device)
agent3.load_state_dict(torch.load('../image/agent_{}.pt'.format('MPC'), map_location=torch.device('cpu')))



# Plot culmulative reward
def plot_creward():
    name1 = dataFilePath + "final_test_EDF.npy"
    name2 = dataFilePath + "final_test_LLF.npy"
    name3 = dataFilePath + "final_test_MPC.npy"

    with open(name1, 'rb') as f:
        EDF = np.load(f)

    with open(name2, 'rb') as f:
        LLF = np.load(f)

    with open(name3, 'rb') as f:
        MPC = np.load(f)

        # plot cumulative reward trend
    plt.figure('culmulative_reward')
    plt.plot(range(len(EDF) - 1), EDF[1:], label='EDF')
    plt.plot(range(len(LLF) - 1), LLF[1:], label='LLF')
    plt.plot(range(len(MPC) - 1), MPC[1:], label='MPC')
    plt.legend()
    plt.xlabel('episode')
    plt.ylabel('culmulative reward')
    plt.title('1000 testing performance of deterministic optimizers')
    plt.savefig(dataFilePath + '1000_testing_performance_comparison.png')
    plt.close('all')



# count(1) returns 1, 2, 3 ...
# test LLF
for i_episode in itertools.count(1):
    episode_reward = 0
    episode_steps = 0
    done = False
    # reset to time (0-24) equal to the first EV errical time, not 0!
    state = env1.reset(dataDirectory, isTrain=False)

    '''
    The while loop will break if and only if: 
    1. episode is finished (one day charging event) 
    or 2. iteration reaches num_steps.
    '''
    while not done:
        # generate action
        data = state.astype(np.float32)
        data = torch.from_numpy(data).type(torch.float32)
        data = data.to(device)
        output = agent1(data)

        action = output.detach().numpy().ravel()

        '''
        Learning Algorithms Should be Invoked Here
        
        Action above is made by random sampling from action space (for space detail see EV_env) ONLY FOR DEMOSTRATION.
        
        Action needs to be taken by training a policy from the algorithm.
        '''

        next_state, reward, done, _, refined_act = env1.step(action)  # Step

        # record cumulative reward from the learning process
        episode_reward += reward
        # Update state to next state
        state = next_state
    # Print episode iterating information
    print("----------------------------------------")
    print("Test culmulative reward for LLF in day {}: {}".format(i_episode ,round(episode_reward, 2)))
    print("----------------------------------------")
    if i_episode > testing_num_days:
        testname = dataFilePath + "final_test_LLF.npy"
        with open(testname, 'wb') as f:
            np.save(f, env1.reward_vec)    
        break    
env1.close()    

# test EDF
for i_episode in itertools.count(1):
    episode_reward = 0
    episode_steps = 0
    done = False
    # reset to time (0-24) equal to the first EV errical time, not 0!
    state = env2.reset(dataDirectory, isTrain=False)

    '''
    The while loop will break if and only if: 
    1. episode is finished (one day charging event) 
    or 2. iteration reaches num_steps.
    '''
    while not done:
        # generate action
        data = state.astype(np.float32)
        data = torch.from_numpy(data).type(torch.float32)
        data = data.to(device)
        output = agent2(data)

        action = output.detach().numpy().ravel()

        '''
        Learning Algorithms Should be Invoked Here
        
        Action above is made by random sampling from action space (for space detail see EV_env) ONLY FOR DEMOSTRATION.
        
        Action needs to be taken by training a policy from the algorithm.
        '''

        next_state, reward, done, _, refined_act = env2.step(action)  # Step

        # record cumulative reward from the learning process
        episode_reward += reward
        # Update state to next state
        state = next_state
    # Print episode iterating information
    print("----------------------------------------")
    print("Test culmulative reward for EDF in day {}: {}".format(i_episode ,round(episode_reward, 2)))
    print("----------------------------------------")
    if i_episode > testing_num_days:
        testname = dataFilePath + "final_test_EDF.npy"
        with open(testname, 'wb') as f:
            np.save(f, env2.reward_vec)    
        break    
env2.close()    



# test MPC
for i_episode in itertools.count(1):
    episode_reward = 0
    episode_steps = 0
    done = False
    # reset to time (0-24) equal to the first EV errical time, not 0!
    state = env3.reset(dataDirectory, isTrain=False)

    '''
    The while loop will break if and only if: 
    1. episode is finished (one day charging event) 
    or 2. iteration reaches num_steps.
    '''
    while not done:
        # generate action
        data = state.astype(np.float32)
        data = torch.from_numpy(data).type(torch.float32)
        data = data.to(device)
        output = agent3(data)

        action = output.detach().numpy().ravel()

        '''
        Learning Algorithms Should be Invoked Here
        
        Action above is made by random sampling from action space (for space detail see EV_env) ONLY FOR DEMOSTRATION.
        
        Action needs to be taken by training a policy from the algorithm.
        '''

        next_state, reward, done, _, refined_act = env3.step(action)  # Step

        # record cumulative reward from the learning process
        episode_reward += reward
        # Update state to next state
        state = next_state
    # Print episode iterating information
    print("----------------------------------------")
    print("Test culmulative reward for MPC in day {}: {}".format(i_episode ,round(episode_reward, 2)))
    print("----------------------------------------")
    if i_episode > testing_num_days:
        testname = dataFilePath + "final_test_MPC.npy"
        with open(testname, 'wb') as f:
            np.save(f, env3.reward_vec)    
        break    
env3.close()  
    
    
# plot the training loss against espisode
if plot_loss:
    plot_creward()










