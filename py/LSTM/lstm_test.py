import argparse
import gym
import gym_EV
import numpy as np
import numpy.linalg as LA
import itertools
import torch
from datetime import datetime
from datetime import timedelta
import matplotlib.pyplot as plt
from gym_EV.envs.reward_functions import *
# import MPC modules
from MPC.objective_functions import *
from MPC.network import *
from MPC.mpc import *
# import lstm modules
from lstm_hyper import *
from lstm import *
# import utils modules
from utils import *
from tensorboardX import SummaryWriter



########################################################
#
# Executive function to run the experiments
#
########################################################    
def main(taskName,
         maxEv = 120, 
         maxRate = 6, 
         maxCapacity = 20, 
         phasePartition = [40, 40],
         reward = [RewardComponent(l2_norm_reward, 1), RewardComponent(deadline_penalty, 0)], 
         objective = [ObjectiveComponent(l1_aggregate_power)], 
         phaseSelection = True, 
         nStep = 0):
    
    ########################################################
    #
    # Initialize experiment variables
    #
    ########################################################    
    gymArgs.MAX_EV = maxEv
    gymArgs.MAX_RATE = maxRate
    netArgs.MAX_CAPACITY = maxCapacity
    netArgs.PHASE_PARTITION = phasePartition
    # Define reward functions
    REWARD_FUNCTION = reward 
    #OBJECTIVE = [ObjectiveComponent(l2_aggregate_power)]
    OBJECTIVE = objective
    dataArgs.PHASE_SELECT = phaseSelection
    lstmArgs.nStep = nStep
    
    ########################################################
    #
    # Initialize tensor board writer
    #
    ########################################################     
    log_data_dir = 'runs/{}_EV={}_RATE={}_CAP={}_NSTEP={}/'.format(
        datetime.now().strftime("%Y%m%d_%H%M%S"), gymArgs.MAX_EV, gymArgs.MAX_RATE, 
        netArgs.MAX_CAPACITY, lstmArgs.nStep)      
    #TesnorboardX
    writerd = SummaryWriter(log_dir=log_data_dir)    
    
    ########################################################
    #
    # Initialize environment
    #
    ########################################################      
    env = gym.make(gymArgs.ENV_NAME)
    env.__init__(dataArgs.END_TRAIN, 
                 dataArgs.END_TEST, 
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
    
    ########################################################
    #
    # Construct charging network
    #
    ######################################################## 
    # construct ev network
    NETWORK = Network(max_ev = gymArgs.MAX_EV, 
                max_rate = gymArgs.MAX_RATE, 
                max_capacity = netArgs.MAX_CAPACITY, 
                turning_ratio = netArgs.TURN_RATIO, 
                phase_partition = netArgs.PHASE_PARTITION,
                constraint_type = netArgs.CONSTRAINT_TYPE)
    # obtain constraint matrix and upper limite and phase information from EV network
    INFRASTRUCTURE = NETWORK.infrastructure_info()
    ########################################################
    #
    # Define objective function and MPC
    #
    ######################################################## 
    # initialize mpc algorithm
    A = AdaptiveChargingOptimization(INFRASTRUCTURE, 
                                     OBJECTIVE, 
                                     max_ev = gymArgs.MAX_EV, 
                                     max_rate = gymArgs.MAX_RATE, 
                                     max_capacity = netArgs.MAX_CAPACITY, 
                                     constraint_type = netArgs.CONSTRAINT_TYPE)
    
    ########################################################
    #
    # Preprocess data and initialize hyperparameters
    #
    ########################################################  
    # construct LSTM model, optimizer. loss function is MSE as default
    model_AB = LSTM(input_size=lstmArgs.numVariables, hidden_layer_size=50, output_size=lstmArgs.numVariables)
    optim_AB = torch.optim.Adam(model_AB.parameters(), lr=0.001)    
    model_BC = LSTM(input_size=lstmArgs.numVariables, hidden_layer_size=50, output_size=lstmArgs.numVariables)
    optim_BC = torch.optim.Adam(model_BC.parameters(), lr=0.001)    
    model_CA = LSTM(input_size=lstmArgs.numVariables, hidden_layer_size=50, output_size=lstmArgs.numVariables)
    optim_CA = torch.optim.Adam(model_CA.parameters(), lr=0.001)    
    model_ALL = LSTM(input_size=lstmArgs.numVariables, hidden_layer_size=50, output_size=lstmArgs.numVariables)
    optim_ALL = torch.optim.Adam(model_ALL.parameters(), lr=0.001)    
    
    ########################################################
    #
    # Load learning agents
    #
    ########################################################  
    # generate ACN data in LSTM format
    lstmData = LSTM_DATA(dataArgs.START, 
                         dataArgs.END_TRAIN, 
                         dataArgs.END_TEST, 
                         windowSize = lstmArgs.windowSize, 
                         phaseSelection = dataArgs.PHASE_SELECT)
    # initialize past charging session
    env.chargingSessions = np.append(lstmData.dataTrain, np.array([lstmData.phaseTrain]).T, axis = 1)
    # upload trained model
    model_AB, optim_AB = load_model(model_AB, optim_AB, PATH = "./runs/model/LSTM_AB.pt")
    model_BC, optim_BC = load_model(model_BC, optim_BC, PATH = "./runs/model/LSTM_BC.pt")
    model_CA, optim_CA = load_model(model_CA, optim_CA, PATH = "./runs/model/LSTM_CA.pt")       
    model_ALL, optim_ALL = load_model(model_ALL, optim_ALL, PATH = "./runs/model/LSTM_ALL.pt")     
    
    ########################################################
    #
    # Start episodic training loop
    #
    ########################################################        
    # Training Loop
    total_numsteps = 0
    # initilaize offline reward
    offlineReward = []
    # count(1) returns 1, 2, 3 ...
    for i_episode in itertools.count(1):
        episode_steps = 0
        done = False
        # reset to time (0-24) equal to the first EV errical time, not 0!
        state = env.reset()
        offlineSessions = get_daily_offline_sessions(env.data)
        offlineReward.append((LA.norm(A.solve(offlineSessions), axis = 0)).sum())
        # obtain optimal action for current time step from MPC
        print("Episode: (day) {0}, optimal offline episodic reward: {1}".format(i_episode, offlineReward[-1]))  
        
        ########################################################
        #
        # Start state evolution with an episode
        #
        ########################################################  
        
        while not done:
            if dataArgs.PHASE_SELECT == False:
                activeSessions = get_optimizing_sessions(env, lstmData, model_ALL, phaseType = None)
            else:
                activeSessions = np.array([])
                sessionsByPhase = [get_optimizing_sessions(env, lstmData, model_AB, phaseType = -1), 
                                                 get_optimizing_sessions(env, lstmData, model_BC, phaseType = 0), 
                                                 get_optimizing_sessions(env, lstmData, model_CA, phaseType = 1)]
                for sessions in sessionsByPhase:
                    if sessions.size != 0:
                        if activeSessions.size == 0:
                            activeSessions = sessions
                        else:
                            activeSessions = np.append(activeSessions, sessions, axis = 0)
            #print(activeSessions)
            # obtain optimal action for current time step from MPC
            action = np.array([float((str(i))) for i in A.solve(activeSessions)[:,0]]) # covert to float ndarray 
            # Print random action
            #print("Episode  ({}): episode step {} taking action: {}".format(i_episode, episode_steps, action))
            
            next_state, reward, done, info = env.step(action) # Step
            # record number of steps taken within the current episode
            episode_steps += 1
            # record number of steps taken in the whole learning process
            total_numsteps += 1
            # Update state to next state
            state = next_state
            # Print current state information
            #print("Next State: {} || Reward: {} || New EVs : {}".format(state, reward, info)) 
       
        ########################################################
        #
        # Summarize episodic results and start evaluation
        #
        ########################################################        
        # save trained data and neural network parameters
        trainname = log_data_dir + taskName +".npy"
        trainname1 = log_data_dir + "offLine_" + taskName +".npy"
        # .npy version
        with open(trainname, 'wb') as f:
            np.save(f, env.reward_vec)
        
        with open(trainname1, 'wb') as f:
            np.save(f, offlineReward)            
        # Print episode iterating information
        print("Episode: (day) {}, total numsteps: {}, episode steps: {}, reward: {}".format(i_episode, total_numsteps, episode_steps, round(env.cul_reward, 2)))    
        # finish learning if the maximum steps of state evulution has been reached
        if i_episode >= expArgs.TRAIN_EPISODES:
            break
    env.close()
    return [offlineReward, env.cul_reward]

main('MPC_L1_Step=50', 
     maxEv = 120,  
     maxRate = 6,  
     maxCapacity = 20, 
     phasePartition = [40, 40],  
     reward = [RewardComponent(l2_norm_reward, 1), RewardComponent(deadline_penalty, 100)],
     objective = [ObjectiveComponent(l1_aggregate_power)],  
     phaseSelection = True,  
     nStep = 50)

main('MPC_L1_Step=0', 
     maxEv = 120,  
     maxRate = 6,  
     maxCapacity = 20, 
     phasePartition = [40, 40],  
     reward = [RewardComponent(l2_norm_reward, 1), RewardComponent(deadline_penalty, 100)],
     objective = [ObjectiveComponent(l1_aggregate_power)],  
     phaseSelection = True,  
     nStep = 0)

main('MPC_QC_Step=0', 
     maxEv = 120,  
     maxRate = 6,  
     maxCapacity = 20, 
     phasePartition = [40, 40],  
     reward = [RewardComponent(l2_norm_reward, 1), RewardComponent(deadline_penalty, 100)],
     objective = [ObjectiveComponent(quick_charge)],  
     phaseSelection = True,  
     nStep = 0)


#plot_whole_folders(DATA_PATH, xlabel = 'episode', ylabel = 'culmulative reward',  low = 0, up = 1)