import pandas as pd
import numpy as np
from collections import OrderedDict
import warnings
import itertools

class Network:
    """
    Configure virtural three-phase network of EV station.
    
    Args:
    max_ev (int): maximum EV slots (number of EVSEs) in the system;
    maxRateVec (float): vector of the maximum power assignment that one EVSE can deliver;
    maxCapacity (float): maximum capacity from the constant power source;
    turning_ratio (int): step-down transformers turning ratio;
    phase_partition (List[int, int]): two-dimensioned vector, the first element of which determines number of EVSEs in AB line, the second of which determines that in BC line.
    constraint_ytpe (str): type of constraints (typs of  power network) currently supports linear and SOC network.
    """
    def __init__(self, 
                 max_ev = 5, 
                 maxRateVec = 5 * [3], 
                 offsetCapacity = 10, 
                 turning_ratio = 4, 
                 phase_partition = [2, 2], 
                 constraint_type = 'SOC'):
        self.max_ev = max_ev
        self.maxRateVec = np.array(maxRateVec)
        self.offsetCapacity = offsetCapacity
        self.turning_ratio = turning_ratio
        self.phase_partition = phase_partition
        self.constraint_type = constraint_type    
        self.infrastructure = self.infrastructure_info()
    


    def get_capacity_constraints(self, externalCapacity):
        ''' Get current capacity constraints based on current external capacity and constraint type. '''
        if externalCapacity < 0:
            raise ValueError("External Capacity must be positive: {0}".format(externalCapacity))       
        if self.constraint_type == 'LINEAR':
            bu = self.offsetCapacity + externalCapacity
        elif self.constraint_type == 'SOC':
            # Upper bound of inequilty
            bu = np.zeros(6)
            bu[0:3] = self.offsetCapacity + externalCapacity
            bu[3:6] = (self.offsetCapacity + externalCapacity) * (np.sqrt(3) / 4)  
        else:
            raise ValueError(
                'Invalid infrastructure constraint type: {0}. Valid options are SOC or AFFINE.'.format(self.constraint_type))                    
        return bu
    
    def get_evse_constraints(self, externalRateVec):
        '''get current real evse constraints based on the maximum EVSE power and maximum EV power intake'''
        if len(externalRateVec) != self.max_ev:    
            raise ValueError("External Rate must be at length of {0} but {1}".format(self.max_ev, len(externalRateVec)))    
        return np.clip(np.array(externalRateVec), a_min = 0, a_max = self.maxRateVec).astype(float)
                
    def infrastructure_info(self):
        """
        Obtain information of simulated network. 
        If it is linear, the only network constratint: max_capacity is returned;
        if it is SOC, output includes: constraint matrix,
        phase vector for all EVSEs and three-phase partitioned points.
        
        Return:
        infrastructure (Dict{'constraint_matrix', 'constraint_limits', 'evse_limits', 'phases', 'phaseid'}): 
        infrastructure information from the function infrastructure_info.
        """
        if self.constraint_type == "LINEAR":
            return {}
        elif self.constraint_type == 'SOC':
            # intialize Constraint Set
            AB = np.zeros(self.max_ev)
            BC = np.zeros(self.max_ev)
            CA = np.zeros(self.max_ev)
            # approximate equipartition on evse into three phase
            p1 = int(self.phase_partition[0])
            p2 = int(self.phase_partition[1])
            # update constraint matrix
            AB[0 : p1] = 1
            BC[p1 : p2 + p1] = 1
            CA[p2 + p1 : ] = 1
            # update phases
            pa = np.zeros(self.max_ev)
            pa[0 : p1] = 30
            pa[p1 : p2 + p1] = -90
            pa[p2 + p1 : ] = 150        
            # Define intermediate currents
            I3a = AB - CA
            I3b = BC - AB
            I3c = CA - BC
            I2a = (1 / self.turning_ratio) * (I3a - I3c)
            I2b = (1 / self.turning_ratio) * (I3b - I3a)
            I2c = (1 / self.turning_ratio) * (I3c - I3b)  
            return {'constraint_matrix' : np.array([I3a, I3b, I3c, I2a, I2b, I2c]), 'phases': pa, 'phaseid': [p1, p2]}
        else:
            raise ValueError(
                'Invalid infrastructure constraint type: {0}. Valid options are SOC or AFFINE.'.format(self.constraint_type))                
        

    
    
    def isFeasible(self, action, externalCapacity, decimal):
        '''
        Given an action vector, check if this action is feasible for the three-phase network with given infrastructure information。
        
        Args:
        infrastructure (Dict{'constraint_matrix', 'constraint_limits', 'phases', 'phaseid'}): infrastructure information from functino infrastructure_info.
        action (np.array) : rate allocation vector be be checked its feasibility.
        
        Return:
        (Bool) : True if it's feasible and False if it's not.
        ''' 
        if (self.constraint_type == "LINEAR"):
            return np.sum(action) <= self.get_capacity_constraints(externalCapacity)
        elif (self.constraint_type == "SOC"):
            phase_in_rad = np.deg2rad(self.infrastructure['phases'])
            cAction = np.stack([action * np.cos(phase_in_rad), action * np.sin(phase_in_rad)])
            absAction = np.linalg.norm(self.infrastructure['constraint_matrix']  @ cAction.T, axis = 1)    
            #print('{0} vs {1}'.format(absAction, self.get_capacity_constraints(externalCapacity)))
            return np.all(np.round(absAction, decimal) <= np.round(self.get_capacity_constraints(externalCapacity), decimal))      
    
    
    def RoundRobinMatrix(self, actionMatrix, externalCapacitySeries, externalRateVecMatrix, decrement = 0.01):
        '''
        Action justification algorithm (matrix-wise) that equally decrease power rate until all constraints of the infrastructure are satisfied.
        
        Args:
        
        action (np.array) : rate allocation vector be be checked its feasibility.
        externalCapacitySeries (np.array(float)): external capacity series for each action is taken.
        externalRateVec (np.array(n_Ev * action.shape[1])): a n_EV dimensioned matrix indicating the max rate intake of each EVSE at all time horizo
        
        Return:
        raction (np.array): Refined action that is enforced within the feasible set.
        '''       
        if len(actionMatrix.shape) != 2:
            return actionMatrix      
        
        if actionMatrix.shape[1] != len(externalCapacitySeries):
            raise ValueError('Dimension Mismatch Between actionMatrix {0} and externalCapacitySeries {1}.'.format(actionMatrix.shape[1], 
                                                                                                                 len(externalCapacitySeries)))       
        if actionMatrix.shape[1] != externalRateVecMatrix.shape[1]:
            raise ValueError('Dimension Mismatch Between actionMatrix {0} and externalRateVecMatrix {1}.'.format(actionMatrix.shape[1], 
                                                                                                                 externalRateVecMatrix.shape[1]))       
        decimalPlace = int(str(decrement)[::-1].find('.'))
        ractionMatrix = np.zeros(actionMatrix.shape)
        for index in range(len(externalCapacitySeries)):
            # initialize action vector
            ractionMatrix[:, index] = self.RoundRobin(actionMatrix[:, index], 
                                                      externalCapacity = externalCapacitySeries[index], 
                                                      externalRateVec = externalRateVecMatrix[:, index], 
                                                      decrement = decrement)
        return ractionMatrix
    
    def RoundRobin(self, action, externalCapacity = 0, externalRateVec = 5 * [0], decrement = 0.01):
        '''
        Action justification algorithm that equally decrease power rate until all constraints of the infrastructure are satisfied.
        
        Args:
        
        action (np.array) : rate allocation vector be be checked its feasibility.
        externalCapacity (float): external capacity when action is taken.
        externalRateVec (np.array(n_Ev)): a n_EV dimensioned vector indicating the max rate intake of each EVSE
        
        Return:
        raction (np.array): Refined action that is enforced within the feasible set.
        '''
        decimalPlace = int(str(decrement)[::-1].find('.'))
        evseLimits = self.get_evse_constraints(externalRateVec)
        # initialize action vector
        raction = np.clip(action, a_min = 0, a_max = evseLimits).astype(float)
        # count(0) returns 0, 1, 2, 3 ...
        for i in itertools.count(0):
            idx = i % self.max_ev
            if self.isFeasible(raction, externalCapacity, decimalPlace):
                break
            else:
                raction[idx] = np.clip(raction[idx] - decrement, a_min = 0 , a_max = evseLimits[idx])  
                #print(raction)               
        return raction  
           
           
            
########################################################
#
# Run as the main module (eg. for testing).
#
########################################################  
if __name__ == "__main__":  
    # construct ev network
    N = Network(max_ev = 5, 
                 maxRateVec = 5 * [3], 
                 offsetCapacity = 0, 
                 turning_ratio = 4, 
                 phase_partition = [2, 2], 
                 constraint_type = 'SOC')  
    externalRateVec = np.abs(np.random.normal(3, 1, 5))
    externalCapacity = np.abs(np.random.normal(4, 3, 5))
    action = np.abs(np.random.normal(5, 10, (5, 5)))
    #action[:,0] = np.array([0.29, 0.29, 0.29, 0.3, 0.3 ])
    infrastructure = N.infrastructure
    print(infrastructure)
    print(action[:, 0])
    raction = N.RoundRobin(action, externalCapacity = externalCapacity, externalRateVec = externalRateVec)   
    print(raction[:,0])
    raction = N.RoundRobin(action[:, 0], externalCapacity = externalCapacity[0], externalRateVec = externalRateVec)   
    print(raction)
            