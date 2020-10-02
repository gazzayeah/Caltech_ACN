import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import os

DEBUG = 1

def DEBUG_PRINT(string):
    global DEBUG
    if DEBUG:
        print("\n {0} \n\r".format(string))

########################################################
#
# Run as the debugging code (eg. for testing).
#
########################################################  
def online_offline_equivalency(env, startTime, onlineMpcDict, onlineActions, offlineMpcDict, offlineActions):
    onlineMpcDict["session_info"] = np.round(onlineMpcDict["session_info"], 2)
    onlineMpcDict["capacity_constraint"] = np.round(onlineMpcDict["capacity_constraint"], 2)
    onlineActions = np.round(onlineActions, 2)
    
    offlineMpcDict["session_info"] = np.round(offlineMpcDict["session_info"], 2)
    offlineMpcDict["capacity_constraint"] = np.round(offlineMpcDict["capacity_constraint"], 2)
    offlineActions = np.round(offlineActions, 2)    

    offlineIdx = int(np.round((env.time - startTime) / env.time_interval))
    print("Offline Index: {0}\n\r".format(offlineIdx))
    
    print("Current AB Sessions: \n\r {0}".format(env.get_current_active_sessions(phaseType = -1)))
    print("Current BC Sessions: \n\r {0}".format(env.get_current_active_sessions(phaseType = 0)))
    print("Current CA Sessions: \n\r {0}".format(env.get_current_active_sessions(phaseType = 1)))
    
    currentAction = onlineActions
    optimalAction = offlineActions[:, offlineIdx]
    if not np.all(np.round(currentAction) == np.round(optimalAction)):
        print("Action Matching False \n\r")
        print("Normal mpc online action: shape = {0} \n {1} \n\r".format(len(currentAction), currentAction))            
        print("Normal mpc offline action: shape = {0} \n {1} \n\r".format(len(optimalAction), optimalAction))  
        print("Sum of previous offline actions: \n {0} \n\r".format(offlineActions[:, : offlineIdx].sum(axis = 1) * env.time_interval))
        if not np.all(np.round(onlineMpcDict["capacity_constraint"], 2) == np.round(offlineMpcDict["capacity_constraint"][offlineIdx:], 2)):
            print("Capacity Time Series Matching False \n\r")
            print(onlineMpcDict["capacity_constraint"] == offlineMpcDict["capacity_constraint"][offlineIdx:])
            print("Normal mpc online Capacity Series: length = {0} \n {1} \n\r".format(len(onlineMpcDict["capacity_constraint"]), 
                                                                                                        onlineMpcDict["capacity_constraint"]))   
            print("Normal mpc offline Capacity Series: length = {0} \n {1} \n\r".format(len(offlineMpcDict["capacity_constraint"][offlineIdx:]), 
                                                                                                        offlineMpcDict["capacity_constraint"][offlineIdx:]))      
        if not np.all(np.round(onlineMpcDict["session_info"], 2) == np.round(offlineMpcDict["session_info"], 2)):
            print("MPC Session Matching : False \n\r.")
            print("Normal online mpc sessions: length = {0} \n {1} \n\r".format(len(onlineMpcDict["session_info"]), onlineMpcDict["session_info"]))   
            print("Normal offline mpc sessions: length = {0} \n {1} \n\r".format(len(offlineMpcDict["session_info"]), offlineMpcDict["session_info"]))    
        return 0
    return 1
            

