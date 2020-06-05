import pandas as pd
import numpy as np
from collections import OrderedDict
import warnings

class Network:
    """
    Configure virtural three-phase network of EV station
    """
    def __init__(self, max_ev = 5, max_rate = 6, max_capacity = 20, turning_ratio = 4, constraint_type = 'SOC'):
        self.max_ev = max_ev
        self.turning_ratio = turning_ratio
        self.max_capacity = max_capacity
        self.max_rate = max_rate  
        self.constraint_type = constraint_type
    
    def infrastructure_info(self):
        if self.constraint_type == "LINLEAR":
            return {'constraint_limits' : self.max_capacity}
        
        # intialize Constraint Set
        AB = np.zeros(self.max_ev)
        BC = np.zeros(self.max_ev)
        CA = np.zeros(self.max_ev)
        
        # approximate equipartition on evse into three phase
        p1 = int(np.floor(self.max_ev / 3))
        p2 = int(np.ceil(self.max_ev / 3))
        
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
        
        # Upper bound of inequilty
        bu = np.zeros(6)
        bu[0:3] = self.max_capacity
        bu[3:6] = int(self.max_capacity * (np.sqrt(3) / 4))
        
        return {'constraint_matrix' : np.array([I3a, I3b, I3c, I2a, I2b, I2c]), 'constraint_limits' : bu, 'phases': pa, 'phaseid': [p1, p2]}
