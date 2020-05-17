import argparse
import datetime
import gym
import gym_EV
import numpy as np
import itertools
import torch
from sac import SAC
from tensorboardX import SummaryWriter
from normalized_actions import NormalizedActions
from replay_memory import ReplayMemory
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()

MAX_EV = 5
MAX_LEVELS = 6
MAX_CAPACITY = 10

parser = argparse.ArgumentParser(description='PyTorch REINFORCE example')

parser.add_argument('--env-name', default="EV-v0",
                    help='name of the environment to run')

parser.add_argument('--policy', default="Deterministic",
                    help='algorithm to use: Gaussian | Deterministic')

parser.add_argument('--eval', type=bool, default=True,
                    help='Evaluates a policy a policy every 10 episode (default:True)')

parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor for reward (default: 0.99)')

parser.add_argument('--tau', type=float, default=0.005, metavar='G',
                    help='target smoothing coefficient(τ) (default: 0.005)')

parser.add_argument('--lr', type=float, default=0.0003, metavar='G',
                    help='learning rate (default: 0.0003)')

parser.add_argument('--alpha', type=float, default=0.05, metavar='G',
                    help='Temperature parameter α determines the relative importance of the entropy term against the reward (default: 0.2)')

parser.add_argument('--automatic_entropy_tuning', type=bool, default=False, metavar='G',
                    help='Temperature parameter α automaically adjusted.')

parser.add_argument('--seed', type=int, default=456, metavar='N',
                    help='random seed (default: 456)')

parser.add_argument('--batch_size', type=int, default=256, metavar='N',
                    help='batch size (default: 256)')

parser.add_argument('--num_steps', type=int, default=1000001, metavar='N',
                    help='maximum number of steps (default: 1000000)')

parser.add_argument('--hidden_size', type=int, default=256, metavar='N',
                    help='hidden size (default: 256)')

parser.add_argument('--updates_per_step', type=int, default=1, metavar='N',
                    help='model updates per simulator step (default: 1)')

parser.add_argument('--start_steps', type=int, default=2000, metavar='N',
                    help='Steps before which samples random actions (default: 10000)')

parser.add_argument('--target_update_interval', type=int, default=1, metavar='N',
                    help='Value target update per no. of updates per step (default: 1)')

parser.add_argument('--replay_size', type=int, default=1000000, metavar='N',
                    help='size of replay buffer (default: 10000000)')

parser.add_argument('--cuda', action="store_true",
                    help='run on CUDA (default: False)')

args = parser.parse_args()

# Environment
# Removing Normalized Actions. 
# Another way to use it = actions * env.action_space.high[0] -> (https://github.com/sfujim/TD3). This does the same thing.
# (or add env._max_episode_steps to normalized_actions.py)
env = gym.make(args.env_name)
env.__init__(max_ev = MAX_EV, number_level = MAX_LEVELS, max_capacity= MAX_CAPACITY)


# Plant random seeds for customized initial state. This prevents wired thing from happening
torch.manual_seed(args.seed)
np.random.seed(args.seed)
env.seed(args.seed)

# Agent
agent = SAC(env.observation_space.shape[0], env.action_space, args)

log_folder_dir = 'runs/{}_SAC_{}_{}_{}_EV={}_LVL={}'.format(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"), args.env_name,
                                                             args.policy, "autotune" if args.automatic_entropy_tuning else "", MAX_EV, MAX_LEVELS)
#TesnorboardX
writer = SummaryWriter(log_dir=log_folder_dir)

# Memory
memory = ReplayMemory(args.replay_size)

# Training Loop
total_numsteps = 0
updates = 0

for i_episode in itertools.count(1):
    episode_reward = 0
    episode_steps = 0
    done = False
    # reset to time (0-24) equal to the first EV errical time, not 0!
    state = env.reset(isTrain=True)
    
    
    while not done:
        if args.start_steps > total_numsteps:
            action = env.action_space.sample()  # Sample random action
        else:
            action = agent.select_action(state)  # Sample action from policy
        
        # sac can be fired only after replay buffer is full -- enough data in the batch
        if len(memory) > args.batch_size:
            print(memory)
            # Number of updates per step in environment
            for i in range(args.updates_per_step):
                # Update parameters of all the networks
                critic_1_loss, critic_2_loss, policy_loss, ent_loss, alpha = agent.update_parameters(memory, args.batch_size, updates)

                writer.add_scalar('loss/critic_1', critic_1_loss, updates)
                writer.add_scalar('loss/critic_2', critic_2_loss, updates)
                writer.add_scalar('loss/policy', policy_loss, updates)
                writer.add_scalar('loss/entropy_loss', ent_loss, updates)
                writer.add_scalar('entropy_temprature/alpha', alpha, updates)
                updates += 1

        next_state, reward, done, _, refined_act = env.step(action) # Step
        episode_steps += 1
        total_numsteps += 1
        episode_reward += reward

        # Ignore the "done" signal if it comes from hitting the time horizon.
        # (https://github.com/openai/spinningup/blob/master/spinup/algos/sac/sac.py)
        mask = 1 if episode_steps == env._max_episode_steps else float(not done)

        memory.push(state, action, reward, next_state, mask) # Append transition to memory

        state = next_state
    print("Total espisode steps in episode (" + str(i_episode) + "): " + str(episode_steps))
    if total_numsteps > args.num_steps:
        break

    writer.add_scalar('reward/train', episode_reward, i_episode)
    print("Episode: {}, total numsteps: {}, episode steps: {}, reward: {}".format(i_episode, total_numsteps, episode_steps, round(episode_reward, 2)))

    # Show performance
    if i_episode % 10 == 0 and args.eval == True:
        state = env.reset(isTrain=False)
        episode_reward = 0
        done = False
        test_time = [env.time]

        while not done:
            action = agent.select_action(state)
            next_state, reward, done, _, refined_act = env.step(action)
            episode_reward += reward
            test_time.append(env.time)
            state = next_state

        plt.figure()
        remained_power = np.array(env.charging_result)
        initial_power = np.array(env.initial_bat)
        charged_power = initial_power - remained_power
        ind = range(len(remained_power))
        p1 = plt.bar(ind, remained_power)
        p2 = plt.bar(ind, charged_power, bottom=remained_power)
        plt.legend((p1[0], p2[0]), ('Remained', 'Charged'))
        plt.savefig(log_folder_dir+'/episode='+str(i_episode)+'_remaining_power.png')

        plt.close('all')

        writer.add_scalar('reward/test', episode_reward, i_episode)

        print("----------------------------------------")
        print("Test Episode: {}, reward: {}".format(i_episode, round(episode_reward, 2)))
        print("----------------------------------------")
        '''

env.close()

