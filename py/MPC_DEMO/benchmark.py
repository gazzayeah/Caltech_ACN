import numpy as np

def LLFEDF(stateDict, infrastructure, max_ev, max_rate, max_capacity, constraint_type, algorithm = 'LLF'):
    # initialize action vector
    action = np.zeros(max_ev)

    maxRateVec = np.ones(max_ev) * max_rate
    if algorithm == 'LLF':
        Priority = np.argsort(stateDict['remain_time'] - stateDict['remain_energy'] / maxRateVec)
    elif algorithm == 'EDF':
        Priority = np.argsort(stateDict['remain_time'])
     # Sample random action
    for i in Priority:  
        action[i] = max_rate
        if (constraint_type == "LINEAR") and (np.sum(action) > max_capacity):
            action[i] +=  max_capacity - np.sum(action)
            break
        elif (constraint_type == "SOC"):
            phase_in_rad = np.deg2rad(infrastructure['phases'])
            while True:
                cAction = np.stack([action * np.cos(phase_in_rad), action * np.sin(phase_in_rad)])
                absAction = np.linalg.norm(infrastructure['constraint_matrix']  @ cAction.T, axis = 1)    
                if np.all( absAction <= infrastructure['constraint_limits']):
                    break
                else:
                    action[i] += -1     
    return action