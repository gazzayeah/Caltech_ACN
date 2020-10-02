from typing import List, Union
from collections import namedtuple
import numpy as np
import cvxpy as cp
from MPC.objective_functions import *
from MPC.network import *



class InfeasibilityException(Exception):
    pass



class AdaptiveChargingOptimization:
    """ Base class for all MPC based charging algorithms.

    Args:
        network (Network Object): network class representing EV charging network
        objective (List[ObjectiveComponent]): List of components which make up the optimization objective.
        dimension (tuple(1 * 2)): sepcify rows and columns of the optimzing matrix
        period: time interval that new network data is retrieved
        enforce_energy_equality (bool): If True, energy delivered must be equal to energy requested for each EV.
            If False, energy delivered must be less than or equal to request.
        solver (str): Backend solver to use. See CVXPY for available solvers.
    """
    def __init__(self, network, 
                 objective : List[ObjectiveComponent], 
                 period = 0.1, 
                 enforce_energy_equality=False, 
                 solver='ECOS'):
        
        self.network = network
        self.objective_configuration = objective
        self.period = period
        self.enforce_energy_equality = enforce_energy_equality
        self.solver = solver
        self.max_ev   = network.max_ev
        self.maxRateVec   = network.maxRateVec
        self.constraint_type = network.constraint_type
        self.offsetCapacity  = network.offsetCapacity
        self.totalCapacityMatrix = None
        self.mpcInputSessions = None
        self.infrastructure = self.network.infrastructure
        self.accuracy = 5
    
    

    @staticmethod
    def charging_rate_bounds(rates: cp.Variable, active_sessions, maxRateVec, period, accuracy):
        """ Get upper and lower bound constraints for each charging rate.

        Args:
            rates (cp.Variable): cvxpy variable representing all charging rates. Shape should be (N, T) where N is the
                total number of EVSEs in the system and T is the length of the optimization horizon.
            active_sessions (np.array[np.array(evse index, arriving time, duration, energy remaining)]): Two dimensional np.array (N * 4). 
                N represents all current & future EVs. Index of the second dimension are [0]: EVSE index, [1] : current time or arrival time
                [2] : job duration of charging job; [3] : current energy remaining; [4] : maximum charging rate; [5] total capacity measured from PV and constant source
            maxRateVec (np.array(max_ev)): time-varying maximum power intake of the connected EVs.
            period (int): time interval that new network data is retrieved

        Returns:
            List[cp.Constraint]: List of lower bound constraint, upper bound constraint.
        """
        lb, ub = np.zeros(rates.shape), np.zeros(rates.shape)
        activeNum = len(active_sessions)
        for event in range(activeNum):
            idx = int(round(active_sessions[event][0]))
            # quantization of arrival time of current charging event
            startTime = int(round(active_sessions[event][1] / period))
            # quantization of departure time of current charging event
            qtzduration = int(round(active_sessions[event][2] / period))
            lb[idx, startTime : startTime + qtzduration] = 0
            ub[idx, startTime : startTime + qtzduration] = np.round(np.min([active_sessions[event][4], maxRateVec[idx]]), accuracy)      
        # To ensure feasibility, replace upper bound with lower bound when they conflict
        ub[ub < lb] = lb[ub < lb]
        return {'charging_rate_bounds.lb': rates >= lb, 'charging_rate_bounds.ub': rates <= ub}

    @staticmethod
    def energy_constraints(rates: cp.Variable, active_sessions, period, enforce_energy_equality, accuracy):
        """ Get constraints on the energy delivered for each session.

        Args:
            rates (cp.Variable): cvxpy variable representing all charging rates. Shape should be (N, T) where N is the
                total number of EVSEs in the system and T is the length of the optimization horizon.
            active_sessions (np.array[np.array(evse index, arriving time, duration, energy remaining)]): Two dimensional np.array (N * 4). 
                N represents all current & future EVs. Index of the second dimension are [0]: EVSE index, [1] : current time or arrival time
                [2] : job duration of charging job; [3] : current energy remaining; [4] : maximum charging rate; [5] total capacity measured from PV and constant source
            max_rate (float): maximum rate an EVSE can assign
            period (float): time interval that new network data is retrieved
            enforce_energy_equality (bool): If True, energy delivered must be equal to energy requested for each EV.
                If False, energy delivered must be less than or equal to request.

        Returns:
            List[cp.Constraint]: List of energy delivered constraints for each session.
        """
        constraints = {}
        activeNum = len(active_sessions)
        for event in range(activeNum):
            idx = int(round(active_sessions[event][0]))
            # quantization of arrival time of current charging event
            startTime = int(round(active_sessions[event][1] / period))
            # quantization of departure time of current charging event
            qtzduration = int(round(active_sessions[event][2] / period))
            planned_energy = cp.sum(rates[idx, startTime : startTime + qtzduration])
            planned_energy *= period
            constraint_name = f'energy_constraints.{event}'
            if enforce_energy_equality:
                constraints[constraint_name] = planned_energy == np.round(active_sessions[event][3], accuracy)
            else:
                constraints[constraint_name] = planned_energy <= np.round(active_sessions[event][3], accuracy)
        return constraints

    @staticmethod
    def infrastructure_constraint(rates: cp.Variable, totalCapacityMatrix, infrastructure, constraint_type):
        """ Get constraints enforcing infrastructure limits. Type SOC regards charging network as three-phase;
        type LINEAR regards charging network as single phase.

        Args:
            rates (cp.Variable): cvxpy variable representing all charging rates. Shape should be (N, T) where N is the
                total number of EVSEs in the system and T is the length of the optimization horizon.
            totalCapacityMatrix (np.array[num_constraints * optimizing_hrizon]): Two dimensional np.array containing time-varying network constraints at all optimizing timestep. 
                num_constraints represents number of constraint types; optimizing horizon equals to the column dimension of optimizing matrix.
            infrastructure (Dict[np.array, np.array, np.array, np.array): network infrastructure information containing 'constraint_matrix', 'phases' and 'phaseid'.
            constraint_type: currently support SOC and LINEAR

        Returns:
            List[cp.Constraint]: List of constraints, one for each bottleneck in the electrical infrastructure.
        """
        #print("Constraint Matrix : \n {0}".format(totalCapacityMatrix))
        constraints = {}
        if constraint_type == 'SOC':
            if infrastructure['phases'] is None:
                raise ValueError('phases is required when using SOC infrastructure constraints.')
            if totalCapacityMatrix.shape[1] != rates.shape[1]:
                raise ValueError('The second dimension of capacity matrix -- rates mismatch : {0} -- {1}'.format(totalCapacityMatrix.shape[1], rates.shape[1]))
            phase_in_rad = np.deg2rad(infrastructure['phases'])
            for j, v in enumerate(infrastructure['constraint_matrix']):
                a = np.stack([v * np.cos(phase_in_rad), v * np.sin(phase_in_rad)])
                constraint_name = f'infrastructure_constraints.{j}'
                constraints[constraint_name] = cp.norm(a @ rates, axis=0) <= totalCapacityMatrix[j]
        elif constraint_type == 'LINEAR':
            return {'infrastructure_constraints': cp.sum(rates, axis=0) <= totalCapacityMatrix}
        else:
            raise ValueError(
                'Invalid infrastructure constraint type: {0}. Valid options are SOC or AFFINE.'.format(constraint_type))        
        
        return constraints



    def build_objective(self, rates: cp.Variable,  **kwargs):
        """
        Set optimizing objectives by objective functions as inputs.
        
        Args:
        rates (cp.Variable): cvxpy variable representing all charging rates. Shape should be (N, T) where N is the
                total number of EVSEs in the system and T is the length of the optimization horizon.
        
        Returns:
        obj (functions): objective function that takes rates as inputs.
        """
        obj = cp.Constant(0)
        for component in self.objective_configuration:
            obj += component.coefficient * component.function(rates, **kwargs)
        return obj



    def build_problem(self, active_sessions, prev_peak:float=0, **kwargs):
        """ Build parts of the optimization problem including variables, constraints, and objective function.

        Args:
            active_sessions (np.array[np.array(evse index, arriving time, duration, energy remaining)]): Two dimensional np.array (N * 4). 
                N represents all current & future EVs. Index of the second dimension are [0] : index of EVSE connected to [1] : current time or arrival time 
                [2] : job duration of charging job; [3] : current energy remaining [4] maximum power rate intake [5] solar capacity value samples at EV's arrival time.
            prev_peak (float): Previous peak current draw during the current billing period.

        Returns:
            Dict[str: object]:
                'objective' : cvxpy expression for the objective of the optimization problem
                'constraints': list of all constraints for the optimization problem
                'variables': dict mapping variable name to cvxpy Variable.
        """
        # initialize rate variable with spcified dimension
        rates = cp.Variable(shape= (self.max_ev, int(round(np.max(active_sessions[:, 1:3].sum(axis = 1)) / self.period))))
        
        # initialize constraint type
        constraints = {}

        # Rate constraints
        constraints.update(self.charging_rate_bounds(rates, active_sessions, self.maxRateVec, self.period, self.accuracy))

        # Energy Delivered Constraints
        constraints.update(self.energy_constraints(rates, active_sessions, self.period, self.enforce_energy_equality, self.accuracy))

        # Peak Limit
        constraints.update(self.infrastructure_constraint(rates, self.totalCapacityMatrix, self.infrastructure, self.constraint_type))

        # Objective Function
        objective = cp.Maximize(self.build_objective(rates, **kwargs))
        return {'objective': objective,
                'constraints': constraints,
                'variables': {'rates': rates}}

    
    
    def combine_network_constraint_to_sessions(self, sessionInfo):
        '''Combine the constraints from EV profile with the constraints of the network.'''
        outputSession = np.zeros(sessionInfo.shape)
        # add constant capacity belonging to the network
        outputSession[:, 0 : 5] = sessionInfo[:, 0 : 5]
        outputSession[:, 5] = sessionInfo[:, 5]+ self.offsetCapacity
        for idx, event in enumerate(outputSession):
            outputSession[idx, 4] = np.clip(outputSession[idx, 4], a_min = 0, a_max = self.maxRateVec[int(np.round(event[0]))]).astype(float)  
        return outputSession
    
    
    
    def update_total_capacity_matrix(self, capacityConstraint):
        """
        Update self.totalCapacityMatrix based on time-varying capacity and EVSE maximum power intake.
        
        Args: 
        capacityConstraint (np.array(float)): time-varying capacity series from external power source (eg. PV plant);
        """
        externalCapacitySeries = capacityConstraint
        if self.constraint_type == 'SOC':
            matrix = np.array([])
            for externalCapacity in externalCapacitySeries:
                bu = self.network.get_capacity_constraints(externalCapacity)
                if matrix.size == 0:
                    matrix = np.array([bu]).T
                else:
                    matrix = np.append(matrix, np.array([bu]).T, axis = 1)
            self.totalCapacityMatrix = np.round(matrix, self.accuracy)
        elif self.constraint_type == 'LINEAR':
            matrix = []
            for externalCapacity in externalCapacitySeries:
                matrix.append(self.network.get_capacity_constraints(externalCapacity))            
            self.totalCapacityMatrix = np.round(np.array(matrix), self.accuracy)
        else:
            raise ValueError(
                'Invalid infrastructure constraint type: {0}. Valid options are SOC or AFFINE.'.format(constraint_type))               
    
    
    
    def solve(self, mpcDict, prev_peak = 0, verbose: bool = False, **kwargs):
        """ Solve optimization problem to create a schedule of charging rates.

        Args:
            mpcDict (Dict{session_info, capacity_constraint}): information to compute mpc, including current + future EV charging
                sessions and capacity series.
            prev_peak (float): Previous peak current draw during the current billing period.
            verbose (bool): See cp.Problem.solve().

        Returns:
            np.Array: Numpy array of charging rates of shape (N, T) where N is the number of EVSEs in the network and
                T is the length of the optimization horizon. Rows are ordered according to the order of evse_index in
                infrastructure.
        """
        if len(mpcDict['session_info']) == 0 or len(mpcDict['capacity_constraint']) == 0:
            return np.zeros((self.max_ev, 1)) 
        else:
            # total capacity matrix will be uploaded and we won't use mpcDict['capacity_constraint'] for computation
            self.update_total_capacity_matrix(mpcDict['capacity_constraint'])    
            # after this step mpcDict constraint is the combined constraint
            self.mpcInputSessions = self.combine_network_constraint_to_sessions(mpcDict["session_info"])
            # Here we take in arguments which describe the problem and build a problem instance.
            problem_dict = self.build_problem(self.mpcInputSessions, prev_peak, **kwargs)
            prob = cp.Problem(problem_dict['objective'], list(problem_dict['constraints'].values()))
            prob.solve(solver=self.solver, verbose=verbose)
            if prob.status not in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
                raise InfeasibilityException(f'Solve failed with status {prob.status}')
            return problem_dict['variables']['rates'].value
    
    
    
    
########################################################
#
# Run as the main module (eg. for testing).
#
########################################################  
if __name__ == "__main__":  
    np.set_printoptions(formatter={'float_kind':'{:f}'.format})
    # construct ev network
    NETWORK = Network(max_ev = 3, 
                 maxRateVec = 3 * [3], 
                 offsetCapacity = 3, 
                 turning_ratio = 4, 
                 phase_partition = [1, 1], 
                 constraint_type = 'SOC')  
    
    externalRateVec = 3 * [1000]
    
    externalCapacity = np.array([0, 0, 0, 0])
    session = np.array([[0, 0.2, 0.1, 2.3, 3, 12.1], [2, 0, 0.2, 3.4, 2, 20], [1, 0, 0.4, 5, 30, 21]])
    mpcDict = {"capacity_constraint" : externalCapacity, "session_info": session}
    
    infrastructure = NETWORK.infrastructure
    print("Network Infrastructure : \n {0}".format(infrastructure))
    
    OBJECTIVE = [ObjectiveComponent(quick_charge)]
    # initialize mpc algorithm
    A = AdaptiveChargingOptimization(NETWORK, OBJECTIVE)    
    
    action = A.solve(mpcDict)
    print("MPC Output Action : \n {0}".format(action))
    #NETWORK.isFeasible(action, externalCapacity)
    raction = NETWORK.RoundRobin(action, externalCapacity = externalCapacity, externalRateVec = externalRateVec)   
    print(raction == action)    