import argparse
import datetime
import gym
import gym_EV_optimization
import numpy as np
import itertools
import torch
from sac import SAC
from cvxopt import matrix, solvers
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()


def create_row_time_contraint(car, current_time, number_EVs, time_interval):
    row = []
    time = 0
    while 0 <= time <= 24:
        for item in range(number_EVs):
            if time == current_time and item == car:
                row.append(1)
            else:
                row.append(0)
        time += time_interval
    return row


def create_row_charging_contraint(car, number_EVs, time_interval):
    row = []
    time = 0
    while 0 <= time <= 24:
        for item in range(number_EVs):
            if item == car:
                row.append(1)
            else:
                row.append(0)
        time += time_interval
    return row


def create_row_rate_contraint(car, current_time, number_EVs, time_interval):
    row = []
    time = 0
    while 0 <= time <= 24:
        for item in range(number_EVs):
            if item == car and time == current_time:
                row.append(1)
            else:
                row.append(0)
        time += time_interval
    return row


def find_optimal(day):
    file_name = '/Users/tonytiny/Documents/Github/gym-EV_data/real_test/data' + str(day) + '.npy'
    data = np.load(file_name)
    number_EVs = len(data)
    time_interval = 0.5
    # Create matrix A:
    A = []
    G = []
    time = 0
    l = 0
    d = 0
    s = 0
    m = 0

        # Arriving departure constraints
    while 0 <= time <= 24:
        for item in range(number_EVs):
            if data[item, 0] > time or data[item, 0] + data[item, 1] < time:
                row = create_row_time_contraint(item, time, number_EVs, time_interval)
                A.append(row)
                l += 1
        time += time_interval
        # Charging constraints
    for item in range(number_EVs):
        row = create_row_charging_contraint(item, number_EVs, time_interval)
        A.append(row)
        s += 1
        # Rate constraints
    time = 0
    while 0 <= time <= 24:
        for item in range(number_EVs):
            row = create_row_rate_contraint(item, time, number_EVs, time_interval)
            G.append(row)
            d += 1
        time += time_interval
    time = 0
    while 0 <= time <= 24:
        for item in range(number_EVs):
            row = create_row_rate_contraint(item, time, number_EVs, time_interval)
            G.append([-x for x in row])
            m += 1
        time += time_interval

    # Create vector h:
    h =[]
    for item in range(d):
        h.append(6.6)
    for item in range(m):
        h.append(0)

    # Create vector b:
    b =[]
    for item in range(l):
        b.append(0)
    for item in range(number_EVs):
        b.append(data[item, 2])

    # Create vector c:
    time = 0
    c = []
    while 0 <= time <= 24:
        for item in range(number_EVs):
            c.append(1 - time / 24)
        time += time_interval

    A = [[A[j][i] for j in range(len(A))] for i in range(len(A[0]))]
    G = [[G[j][i] for j in range(len(G))] for i in range(len(G[0]))]

    A = matrix(A, tc='d')
    G = matrix(G, tc='d')
    b = matrix(b)
    c = matrix(c)
    h = matrix(h)
    sol = solvers.lp(c=c, G=G, h=h, A=A, b=b)
    charging_sol = sol['x']
    charging = []
    time = 0
    index = 0
    while 0 <= time <= 24:
        aggregate_power = 0
        index = time / time_interval
        for item in range(number_EVs):
            aggregate_power += charging_sol[int(index * number_EVs +  item)]
        charging.append(aggregate_power)
        time += time_interval

    return sol['primal objective'], charging


parser = argparse.ArgumentParser(description='PyTorch REINFORCE example')
parser.add_argument('--env-name', default="EV_optimization-v0",
                    help='name of the environment to run')
parser.add_argument('--policy', default="Gaussian",
                    help='algorithm to use: Gaussian | Deterministic')
parser.add_argument('--eval', type=bool, default=True,
                    help='Evaluates a policy a policy every 10 episode (default:True)')
parser.add_argument('--gamma', type=float, default=0.5, metavar='G',
                    help='discount factor for reward (default: 0.99)')
parser.add_argument('--tau', type=float, default=0.005, metavar='G',
                    help='target smoothing coefficient(τ) (default: 0.005)')
parser.add_argument('--lr', type=float, default=0.0003, metavar='G',
                    help='learning rate (default: 0.0003)')
parser.add_argument('--alpha', type=float, default=0.5, metavar='G',
                    help='Temperature parameter α determines the relative importance of the entropy term against the reward (default: 0.2)')
parser.add_argument('--automatic_entropy_tuning', type=bool, default=False, metavar='G',
                    help='Temperature parameter α automatically adjusted.')
parser.add_argument('--seed', type=int, default=456, metavar='N',
                    help='random seed (default: 456)')
parser.add_argument('--batch_size', type=int, default=256, metavar='N',
                    help='batch size (default: 256)')
parser.add_argument('--num_steps', type=int, default=1000000, metavar='N',
                    help='maximum number of steps (default: 1000000)')
parser.add_argument('--hidden_size', type=int, default=256, metavar='N',
                    help='hidden size (default: 256)')
parser.add_argument('--updates_per_step', type=int, default=1, metavar='N',
                    help='model updates per simulator step (default: 1)')
parser.add_argument('--start_steps', type=int, default=10000, metavar='N',
                    help='Steps sampling random actions (default: 10000)')
parser.add_argument('--target_update_interval', type=int, default=1, metavar='N',
                    help='Value target update per no. of updates per step (default: 1)')
parser.add_argument('--replay_size', type=int, default=1000000, metavar='N',
                    help='size of replay buffer (default: 10000000)')
parser.add_argument('--cuda', action="store_true",
                    help='run on CUDA (default: False)')
args = parser.parse_args()

# Environment

MAX_EV = 54
MAX_LEVELS = 20
MAX_CAPACITY = 150
TUNING = 40

env = gym.make(args.env_name)
torch.manual_seed(args.seed)
np.random.seed(args.seed)
env.__init__(n_EVs = MAX_EV, n_levels = MAX_LEVELS, max_capacity = MAX_CAPACITY, tuning_parameter = TUNING)
env.seed(args.seed)

log_folder_dir = 'runs/test/Tuning={}_{}Updatedpenalty'.format(env.tuning_parameter, datetime.datetime.now().strftime("%Y-%m-%d_%H-%M"))

cost_list = []
offline_cost_list = []
flexibility_list = []
charging_list = []
tracking_list = []
optimal_charging_list = []

# Agent
agent = SAC(env.observation_space.shape[0], env.action_space, args)
agent.load_model("/Users/tonytiny/Documents/GitHub/SAC/models/sac_actor_EV_optimization-v0_Tunning=40","/Users/tonytiny/Documents/GitHub/SAC/models/sac_critic_EV_optimization-v0_Tunning=40")

for day in range(60):
    if day == 1 or day == 12 or day == 22 or day == 52:
        continue
    print(day)

    # receding horizon control

    state = env.reset_for_day_test(day)
    episode_reward = 0
    done = False
    test_time = [env.time]

    test_track_signal = [0, 0]
    test_real_output = [0, 0]
    test_feedback = [np.zeros([1, env.n_levels])]

    tracking_error_list = []
    penalty_list = []

    while not done:

        action = agent.select_action(state)
        next_state, _, done, _, refined_act = env.step(action)
        state = next_state

        tracking_error_list.append(env.tracking_error)
        penalty_list.append(env.penalty)
        test_time.append(env.time)
        test_real_output.append(env.power)
        test_feedback.append(refined_act[-env.n_levels:])
        test_track_signal.append(env.signal)

    cost_list.append(env.cost)
    flexibility_list.append(env.total_flexibility)
    charging_list.append(env.total_charging_error)
    tracking_list.append(env.total_tracking_error)

    feedback_heat = np.zeros([env.n_levels, len(test_feedback)])
    for i, feedback in enumerate(test_feedback):
        feedback_heat[:, i] = feedback
    plt.figure()
    ax = sns.heatmap(feedback_heat)
    plt.title("Heatmap for feedback vectors")
    plt.savefig(log_folder_dir + '_heatmap.png')

    test_real_signal = test_real_output
    plt.figure()
    plt.plot(test_time, test_track_signal[:-1], 'b', label="Tracking Signal")
    plt.plot(test_time, test_real_signal[:-1], 'r', label="Real Output")
    plt.title("Tracking performance")
    plt.savefig(log_folder_dir + '_tracking.png')

    plt.figure()
    remained_power = np.array(env.charging_result)
    initial_power = np.array(env.initial_bat)
    charged_power = initial_power - remained_power
    ind = range(len(remained_power))
    p1 = plt.bar(ind, remained_power)
    p2 = plt.bar(ind, charged_power, bottom=remained_power)
    plt.legend((p1[0], p2[0]), ('Remained', 'Charged'))
    plt.savefig(log_folder_dir + '_remaining_power.png')
    plt.close('all')

    # offline optimal

    offline_optimal, optimal_charging = find_optimal(day)
    # plt.figure()
    # plt.plot(optimal_charging)
    # plt.show()
    offline_cost_list.append(offline_optimal)
    optimal_charging_list.append(optimal_charging)

    name = 'new_results/cost_list_40' + '.npy'
    np.save(name, cost_list)
    # name = 'new_results/offline_cost_list' + '.npy'
    # np.save(name, offline_cost_list)
    name = 'new_results/flexibility_list_40' + '.npy'
    np.save(name, flexibility_list)
    name = 'new_results/charging_list_40' + '.npy'
    np.save(name, charging_list)
    name = 'new_results/tracking_list_40' + '.npy'
    np.save(name, tracking_list)

env.close()

plt.figure()
plt.plot(cost_list)
plt.plot(offline_cost_list)
plt.title("Costs")
plt.savefig(log_folder_dir+'_cost.png')
plt.show()




