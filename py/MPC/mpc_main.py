import argparse
import gym
import MPC.mpc
import gym_EV
import numpy as np
import itertools
import torch
import datetime
from datetime import timedelta
import matplotlib.pyplot as plt
from MPC.test import *
from MPC.utils import *
from mpc_hyper import *
from gym_EV.envs.reward_functions import *
# import MPC modules
from MPC.objective_functions import *
from MPC.network import *
from MPC.mpc import *
import sys
from tensorboardX import SummaryWriter
np.set_printoptions(threshold=sys.maxsize, precision=2, formatter={'float_kind':'{:2f}'.format})



########################################################
#
# Initialize tensor board writer
#
########################################################   
log_data_dir = 'runs/{}_EV={}_RATE={}_CAP={}/'.format(
    datetime.now().strftime("%Y%m%d_%H%M%S"), gymArgs.MAX_EV, netArgs.MAX_RATE_VECTOR[0], 
    netArgs.MAX_EXTERNAL_CAPACITY)      
#TesnorboardX
if not DEBUG:
    writerd = SummaryWriter(log_dir=log_data_dir)    

########################################################
#
# Construct charging network
#
######################################################## 
# construct ev network
NETWORK = Network(max_ev = gymArgs.MAX_EV, 
            maxRateVec = netArgs.MAX_RATE_VECTOR, 
            offsetCapacity = netArgs.MAX_CAPACITY, 
            turning_ratio = netArgs.TURN_RATIO, 
            phase_partition = netArgs.PHASE_PARTITION,
            constraint_type = netArgs.CONSTRAINT_TYPE)

########################################################
#
# Initialize environment
#
########################################################      
# Environment
# Removing Normalized Actions. 
# Another way to use it = actions * env.action_space.high[0] -> (https://github.com/sfujim/TD3). This does the same thing.
# (or add env._max_episode_steps to normalized_actions.py)
REWARD_FUNCTION = [RewardComponent(l1_norm_reward, 1)]
env = gym.make(gymArgs.ENV_NAME)
env.__init__(network = NETWORK, 
             start = dataArgs.END_TRAIN, 
             end = dataArgs.END_TEST, 
             reward = REWARD_FUNCTION, 
             maxExternalCapacity = netArgs.MAX_EXTERNAL_CAPACITY,
             intensity = gymArgs.INTENSITY, 
             phase_selection = dataArgs.PHASE_SELECT, 
             isRandomDate = gymArgs.RANDOM_DATE)
# Plant random seeds for customized initial state. This prevents wired thing from happening
torch.manual_seed(gymArgs.SEED)
np.random.seed(gymArgs.SEED)
env.seed(gymArgs.SEED)



########################################################
#
# Define objective function and MPC
#
######################################################## 
OBJECTIVE = [ObjectiveComponent(quick_charge)]
#OBJECTIVE = [ObjectiveComponent(l1_aggregate_power)]
# initialize mpc algorithm
A = AdaptiveChargingOptimization(NETWORK, OBJECTIVE)


# Training Loop
total_numsteps = 0
# initilaize offline reward
offlineNormalRewards = []
# variable subject to change in the exp
sampleType = "arrivalDelay"
#nStepList = [-1, -1, -1]
nStepList = -1
rewardVec = []
brrVec = []
# count(1) returns 1, 2, 3 ...
for i_episode in itertools.count(1):
    episode_reward = 0
    episode_steps = 0
    done = False
    # reset to time (0-24) equal to the first EV errical time, not 0!
    state = env.reset()
    startTime = env.time
    
    #######################################################
    #
    # Comput Normal-sampling-based offline MPC reward
    #
    #######################################################  
    print("Start Normal-sampling-based Offline Computation: \n\r")
    print("Current Date : {0} -- Time: {1} \n\r".format(env.tmpdate  - timedelta(days = 1)  ,round(env.time, 1)))    
    
    # obtain mpc dictionary from EV gym, this shoud be expected that all constraints are only the EV profile constraints
    offlineMpcDict = env.get_mpc_session_info(nStepList = [-1, -1, -1], sampleType = "arrivalDelay")  
    # check the correction of offlineMpcDict
    DEBUG_PRINT("EV profile Sessions: length = {0} \n {1}".format(len(offlineMpcDict["session_info"]), offlineMpcDict["session_info"]))    
    DEBUG_PRINT("EV profile Capacity Series: length = {0} \n {1}".format(len(offlineMpcDict["capacity_constraint"]), offlineMpcDict["capacity_constraint"])) 
    DEBUG_PRINT("EV profile Max Rate Matrix dimension: length = {0}".format(offlineMpcDict["rate_info"].shape)) 
    startIdx = int(np.round(env.time / env.time_interval))
    endIdx = startIdx + len(offlineMpcDict["capacity_constraint"])
    if len(env.dailyCapacityArray) <= endIdx:
        endIdx = len(env.dailyCapacityArray) 
    solarConstraint = env.dailyCapacityArray[startIdx : endIdx]       
    DEBUG_PRINT("Real Capacity Series: length = {0} \n {1}".format(len(solarConstraint), solarConstraint))    
    # obtain initialized action computed by MPC, in which the network constraints are considered
    offlineActions1 = A.solve(offlineMpcDict, **{"time" : env.time, "period" : 0.1})
    # print updated constraint matrix 
    DEBUG_PRINT("Updated Input Sessions to MPC: \n {0}".format(A.mpcInputSessions))  
    DEBUG_PRINT("First Row of Capacity Matrix: \n {0}".format(A.totalCapacityMatrix[0, :]))
    DEBUG_PRINT("First Column of Capacity Matrix: \n {0}".format(A.totalCapacityMatrix[:, 0]))    
    
    if len(solarConstraint) != 0:
        # Sample random action        
        offlineActions = NETWORK.RoundRobinMatrix(offlineActions1, 
                                           externalCapacitySeries = solarConstraint, 
                                           externalRateVecMatrix = offlineMpcDict["rate_info"])     
    else:
        offlineActions = offlineActions1
    DEBUG_PRINT("Offline Action Regularization Before and After Equal: \n {0}".format(np.all(np.round(offlineActions, 2) == np.round(offlineActions1, 2))))
     
    # Start Computing Reward Function
    offlineDailyReward = 0
    # norm of action vector representing aggregated charging reward
    for component in REWARD_FUNCTION:
        offlineDailyReward += component.coefficient * component.function(offlineActions, [], 0, **component.kwargs)        
    offlineNormalRewards.append(offlineDailyReward)
    # obtain optimal action for current time step from MPC
    print("Episode: (day) {0}, optimal normal-based offline episodic reward: {1}\n\r".format(i_episode, offlineNormalRewards[-1]))    
    
    
    #c = 0
    #######################################################
    #
    # Comput Online Daily MPC reward
    #
    #######################################################  
    print("Start {0}-based Online Computation: \n\r".format(sampleType))
    DEBUG_PRINT("Phase Information: \n AB: {0}; BC:{1}; CA: {2}".format(len(env.data[np.where(env.data[:, 0] == -1)[0]]), 
                                                                        len(env.data[np.where(env.data[:, 0] == 0)[0]]), 
                                                                        len(env.data[np.where(env.data[:, 0] == 1)[0]])))    
    while not done:
        #c += 1
        #if c == 10:
            #break
        print("Current Date : {0} -- Time: {1} \n\r".format(env.tmpdate  - timedelta(days = 1)  ,round(env.time, 1)))    
        # obtain mpc dictionary from EV gym, this shoud be expected that all constraints are only the EV profile constraints
        onlineMpcDict = env.get_mpc_session_info(nStepList = nStepList, sampleType = sampleType) 
        # check the correction of onlineMpcDict
        DEBUG_PRINT("EV profile Sessions: length = {0} \n {1}".format(len(onlineMpcDict["session_info"]), onlineMpcDict["session_info"]))    
        DEBUG_PRINT("EV profile Capacity Series: length = {0} \n {1}".format(len(onlineMpcDict["capacity_constraint"]), onlineMpcDict["capacity_constraint"]))       
        startIdx = int(np.round(env.time / env.time_interval))
        endIdx = startIdx + len(onlineMpcDict["capacity_constraint"])
        if len(env.dailyCapacityArray) <= endIdx:
            endIdx = len(env.dailyCapacityArray) 
        solarConstraint = env.dailyCapacityArray[startIdx : endIdx]   
        DEBUG_PRINT("Real Capacity Series: length = {0} \n {1}".format(len(solarConstraint), solarConstraint))          
        # obtain initialized action computed by MPC, in which the network constraints are considered
        onlineActions= A.solve(onlineMpcDict, **{"time" : env.time, "period" : 0.1})[:,0] # covert to float ndarray    
        brrVec.append(onlineActions.sum())
        # print updated constraint matrix 
        DEBUG_PRINT("Updated Input Sessions to MPC: \n {0}".format(A.mpcInputSessions))  
        DEBUG_PRINT("First Row of Capacity Matrix: \n {0}".format(A.totalCapacityMatrix[0, :]))
        DEBUG_PRINT("First Column of Capacity Matrix: \n {0}".format(A.totalCapacityMatrix[:, 0]))    
        DEBUG_PRINT("Online Round Robin Inputs: \n\r Capacity Match: {0} -- {1} \n\r Rate Vector: {2} \n\r".format(env.externalCapacity, 
                                                                                                                   env.externalCapacity == env.dailyCapacityArray[int(round(env.time / env.time_interval))], 
                                                                                                                   env.externalRateVec))     
        ########################################################
        #
        # Testing Code Session
        #
        ########################################################        
        #if not online_offline_equivalency(env, startTime, onlineMpcDict, onlineActions, offlineMpcDict, offlineActions1):
            #offlineIdx = int(np.round((env.time - startTime) / env.time_interval))
            #break   
                    
        next_state, reward, done, info = env.step(onlineActions) # Step
        #print("New ev arrival information: {0}".format(info))
        # record number of steps taken within the current episode
        episode_steps += 1
        # record number of steps taken in the whole learning process
        total_numsteps += 1
        # record cumulative reward from the learning process
        episode_reward += reward
        rewardVec.append(reward)
        # Update state to next state
        state = next_state
        # Print current state information
        #print("Next State: {} || Reward: {} || New EVs : {}".format(state, reward, info))         
    print("\n Episode: (day) {}, total numsteps: {}, episode steps: {}, reward: {}".format(i_episode, total_numsteps, episode_steps, round(episode_reward, 2)))
    
    
    
    
    # finish learning if the maximum steps of state evulution has been reached
    if i_episode >= expArgs.TRAIN_EPISODES:
        ########################################################
        #
        # Summarize episodic results and start evaluation
        #
        ########################################################            
        if not DEBUG:
            #plt.figure(1)
            #plt.plot(range(expArgs.TRAIN_EPISODES), offlineNormalRewards, label = "Offline Normal")
            #plt.plot(range(expArgs.TRAIN_EPISODES), offlineArrivalRewards, label = "Offline Arrival")
            #plt.plot(range(expArgs.TRAIN_EPISODES), env.reward_vec, label = "Online 0-step {}".format(sampleType))
            #plt.xlabel("Days")
            #plt.ylabel("Culmulative Rewards")
            #plt.legend(loc = 'upper right')
            #plt.title("N Step Perfect Prediction")    
            #plt.show()                
            # save trained data and neural network parameters
            trainname = log_data_dir + "offlineNormalRewards.npy"
            if type(nStepList) != int:
                if sum(nStepList) < 0:
                    label = 1000
                else:
                    label = sum(nStepList) 
            else:
                label = nStepList
            trainname1 = log_data_dir + "online_{0}_step={1}.npy".format(sampleType, label)
            # .npy version
            with open(trainname, 'wb') as f:
                np.save(f, offlineNormalRewards)
                
            with open(trainname1, 'wb') as f:
                np.save(f, env.reward_vec)          
        break    
    
env.close()


