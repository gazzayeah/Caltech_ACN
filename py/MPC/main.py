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




training_num_days = 1000
testing_num_days = 10
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
#writer = SummaryWriter(log_dir=log_folder_dir)

# construct ev network
N = Network(max_ev=MAX_EV, max_rate=MAX_RATE, max_capacity=MAX_CAPACITY, turning_ratio=TURN_RATIO,
            constraint_type=CONSTRAINT_TYPE)

# obtain constraint matrix and upper limite and phase information from EV network
infrastructure = N.infrastructure_info()

# initialize mpc algorithm
A = AdaptiveChargingOptimization(infrastructure, max_ev=MAX_EV, max_rate=MAX_RATE, max_capacity=MAX_CAPACITY,
                                 constraint_type=CONSTRAINT_TYPE)

# initialize the agent model
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
agent = LearningAgent(action_dim=MAX_EV, state_dim=MAX_EV * 2, hidden_dim=MAX_EV).to(device)
optimizer = torch.optim.Adam(agent.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)



# Testing function
def test():
    env1 = gym.make(args.env_name)
    env1.__init__(args.gamma, args.phi, max_ev = MAX_EV, max_rate = MAX_RATE, max_capacity= MAX_CAPACITY, intensity = MAX_INTENSITY)
    env1.seed(args.seed)
    agent.eval()
    episode_reward = 0
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
        output = agent(data)

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
    print("Test culmulative reward for one day: {}".format(round(episode_reward, 2)))
    print("----------------------------------------")
    env1.close()
    return episode_reward

# Plot culmulative reward
def plot_creward(T = 'train'):
    name1 = dataFilePath + "{}_creward_EDF.npy".format(T)
    name2 = dataFilePath + "{}_creward_LLF.npy".format(T)
    name3 = dataFilePath + "{}_creward_MPC.npy".format(T)

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
    plt.title('{} performance of deterministic optimizers'.format(T))
    plt.savefig(dataFilePath + '{}_performance_comparison.png'.format(T))
    plt.close('all')

def plot_test_creward(algorithm):
    name = dataFilePath + "test_creward_{}.npy".format(algorithm)
    with open(name, 'rb') as f:
        TEST = np.load(f)
        # plot cumulative reward trend
    plt.figure('culmulative_reward_test')
    plt.plot(range(len(TEST)), TEST)
    plt.legend()
    plt.xlabel('per_{}_episode'.format(testing_num_days))
    plt.ylabel('culmulative reward')
    plt.title('Performance of Trained Neural Network on Testing Data')
    plt.show()
    plt.savefig(dataFilePath + 'train_performance_comparison.png')
    # plt.close('all')        

def plot_episode_loss(episode_loss):
    loss = [l / sum(episode_loss) for l in episode_loss]
    plt.plot(range(1, len(loss) + 1), loss)
    plt.legend(loc='best')
    plt.xlabel('Episode')
    plt.ylabel('Average episode Loss')
    plt.title('Average training episode Loss')
    plt.savefig('../image/loss.png', bbox_inches='tight')












# count(1) returns 1, 2, 3 ...
# training
episode_losses = []
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
            action = A.solve(env.get_active_state)[:,0]
            action = np.array([float((str(i))) for i in action])# covert to float ndarray

        '''
        Agent Learning from MPC
        '''
        data = state.astype(np.float32)
        data = torch.from_numpy(data).type(torch.float32)
        target = action.astype(np.float32)
        target = torch.from_numpy(action).type(torch.float32)

        agent.train()
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()  # Clear the gradient
        output = agent(data)  # Make predictions
        loss = F.mse_loss(output, target)  # Compute loss
        train_loss += loss
        loss.backward()  # Gradient computation
        optimizer.step()  # Perform a single optimization step


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
    episode_losses.append(train_loss)
    # finish learning if the maximum steps of state evulution has been reached
    # run 100 days
    # Print episode iterating information
    print("Episode: (day) {}, total numsteps: {}, episode steps: {}, reward: {}".format(i_episode, total_numsteps, episode_steps, round(episode_reward, 2)))    
    
    # Show performance
    if i_episode % testing_num_days == 0 and args.eval == True:
        '''
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
        '''
        test_creward_vec.append(test())
    
    if i_episode > training_num_days:
        trainname = dataFilePath + "train_creward_{}.npy".format(ACTION)
        with open(trainname, 'wb') as f:
            np.save(f, env.reward_vec[1:])
        testname = dataFilePath + "test_creward_{}.npy".format(ACTION)
        with open(testname, 'wb') as f:
            np.save(f, test_creward_vec)        
        break

env.close()
# plot the training loss against espisode
if plot_loss:
    plot_episode_loss(episode_losses)
    plot_train_creward()
    plot_test_creward(ACTION)
# save updated param of nn
torch.save(agent.state_dict(), "../image/agent_{}.pt".format(ACTION))

#agent.load_state_dict(torch.load('../image/agent_{}.pt'.format(ACTION), map_location=torch.device('cpu')))















