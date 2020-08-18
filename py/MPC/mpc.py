from typing import List, Union
from collections import namedtuple
import numpy as np
import cvxpy as cp
from MPC.objective_functions import *



class InfeasibilityException(Exception):
    pass



class AdaptiveChargingOptimization:
    """ Base class for all MPC based charging algorithms.

    Args:
        infrastructure (Dict[np.array, np.array, np.array, np.array): network infrastructure information, see class infrastructure_info in network.py
        objective (List[ObjectiveComponent]): List of components which make up the optimization objective.
        max_ev: maximum EV network can take (number of EVSEs)
        max_rate: maximum rate an EVSE can assign
        max_capacity: peak rate the network can deliver
        period: time interval that new network data is retrieved
        enforce_energy_equality (bool): If True, energy delivered must be equal to energy requested for each EV.
            If False, energy delivered must be less than or equal to request.
        solver (str): Backend solver to use. See CVXPY for available solvers.
        constraint_type: currently support SOC and LINEAR
    """
    def __init__(self, infrastructure, objective: List[ObjectiveComponent], max_ev = 5, max_rate = 6, max_capacity = 10, period = 0.1,
                 enforce_energy_equality=False, solver='ECOS', constraint_type = 'SOC'):
        self.enforce_energy_equality = enforce_energy_equality
        self.solver = solver
        self.objective_configuration = objective
        self.max_ev   = max_ev
        self.max_rate   = max_rate
        self.max_capacity  = max_capacity
        self.period = period
        self.infrastructure = infrastructure
        self.constraint_type = constraint_type

    @staticmethod
    def charging_rate_bounds(rates: cp.Variable, active_sessions, max_rate, period):
        """ Get upper and lower bound constraints for each charging rate.

        Args:
            rates (cp.Variable): cvxpy variable representing all charging rates. Shape should be (N, T) where N is the
                total number of EVSEs in the system and T is the length of the optimization horizon.
            active_sessions (np.array[np.array(evse index, arriving time, duration, energy remaining)]): Two dimensional np.array (N * 4). 
                N represents all current & future EVs. Index of the second dimension are [0]: EVSE index, [1] : current time or arrival time
                [2] : job duration of charging job; [3] : current energy remaining.
            max_rate (int): maximum rate an EVSE can assign
            period (int): time interval that new network data is retrieved

        Returns:
            List[cp.Constraint]: List of lower bound constraint, upper bound constraint.
        """
        lb, ub = np.zeros(rates.shape), np.zeros(rates.shape)
        activeNum = len(active_sessions)
        for event in range(activeNum):
            idx = active_sessions[event][0].astype(int)
            # quantization of arrival time of current charging event
            startTime = np.ceil(active_sessions[event][1] / period).astype(int)
            # quantization of departure time of current charging event
            qtzduration = np.ceil(active_sessions[event][2] / period).astype(int)
            lb[idx, startTime : startTime + qtzduration] = 0
            ub[idx, startTime : startTime + qtzduration] = max_rate          
            
        # To ensure feasibility, replace upper bound with lower bound when they conflict
        ub[ub < lb] = lb[ub < lb]
        return {'charging_rate_bounds.lb': rates >= lb, 'charging_rate_bounds.ub': rates <= ub}

    @staticmethod
    def energy_constraints(rates: cp.Variable, active_sessions, period, enforce_energy_equality=False):
        """ Get constraints on the energy delivered for each session.

        Args:
            rates (cp.Variable): cvxpy variable representing all charging rates. Shape should be (N, T) where N is the
                total number of EVSEs in the system and T is the length of the optimization horizon.
            active_sessions (np.array[np.array(evse index, arriving time, duration, energy remaining)]): Two dimensional np.array (N * 4). 
                N represents all current & future EVs. Index of the second dimension are [0]: EVSE index, [1] : current time or arrival time
                [2] : job duration of charging job; [3] : current energy remaining.
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
            idx = active_sessions[event][0].astype(int)
           # quantization of arrival time of current charging event
            startTime = np.ceil(active_sessions[event][1] / period).astype(int)
            # quantization of departure time of current charging event
            qtzduration = np.ceil(active_sessions[event][2] / period).astype(int)  
            planned_energy = cp.sum(rates[idx, startTime : startTime + qtzduration])
            planned_energy *= period
            constraint_name = f'energy_constraints.{event}'
            if enforce_energy_equality:
                constraints[constraint_name] = planned_energy == active_sessions[event][3]
            else:
                constraints[constraint_name] = planned_energy <= active_sessions[event][3]
        return constraints

    @staticmethod
    def infrastructure_constraint(rates: cp.Variable, infrastructure, constraint_type):
        """ Get constraints enforcing infrastructure limits. Type SOC regards charging network as three-phase;
        type LINEAR regards charging network as single phase.

        Args:
            rates (cp.Variable): cvxpy variable representing all charging rates. Shape should be (N, T) where N is the
                total number of EVSEs in the system and T is the length of the optimization horizon.
            infrastructure (Dict[np.array, np.array, np.array, np.array): network infrastructure information, see class infrastructure_info in network.py
            constraint_type: currently support SOC and LINEAR

        Returns:
            List[cp.Constraint]: List of constraints, one for each bottleneck in the electrical infrastructure.
        """
        constraints = {}
        if constraint_type == 'SOC':
            if infrastructure['phases'] is None:
                raise ValueError('phases is required when using SOC infrastructure constraints.')
            phase_in_rad = np.deg2rad(infrastructure['phases'])
            for j, v in enumerate(infrastructure['constraint_matrix']):
                a = np.stack([v * np.cos(phase_in_rad), v * np.sin(phase_in_rad)])
                constraint_name = f'infrastructure_constraints.{j}'
                constraints[constraint_name] = cp.norm(a @ rates, axis=0) <= infrastructure['constraint_limits'][j]
        elif constraint_type == 'LINEAR':
            return {'infrastructure_constraints': cp.sum(rates, axis=0) <= infrastructure['constraint_limits']}
        else:
            raise ValueError(
                'Invalid infrastructure constraint type: {0}. Valid options are SOC or AFFINE.'.format(constraint_type))        
        
        return constraints



    def build_objective(self, rates: cp.Variable, **kwargs):
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



    def build_problem(self, active_sessions, infrastructure, constraint_type, prev_peak:float=0, **kwargs):
        """ Build parts of the optimization problem including variables, constraints, and objective function.

        Args:
            active_sessions (np.array[np.array(evse index, arriving time, duration, energy remaining)]): Two dimensional np.array (N * 4). 
                N represents all current & future EVs. Index of the second dimension are [0]: EVSE index, [1] : current time or arrival time
                [2] : job duration of charging job; [3] : current energy remaining.
            infrastructure (Dict[np.array, np.array, np.array, np.array): network infrastructure information, see class infrastructure_info in network.py
            constraint_type: currently support SOC and LINEAR
            prev_peak (float): Previous peak current draw during the current billing period.

        Returns:
            Dict[str: object]:
                'objective' : cvxpy expression for the objective of the optimization problem
                'constraints': list of all constraints for the optimization problem
                'variables': dict mapping variable name to cvxpy Variable.
        """
        # obtain optimization horizon: minimum of 1. maximum daily time-step and 2. maximum departure time-step.
        optimization_horizon = min(max((np.ceil(active_sessions[:, 1:3] / self.period).astype(int)).sum(axis = 1)) , int(24 / self.period))  
        # initialize rate variable
        rates = cp.Variable(shape=(self.max_ev, optimization_horizon))
        # initialize constraint type
        constraints = {}

        # Rate constraints
        constraints.update(self.charging_rate_bounds(rates, active_sessions, self.max_rate, self.period))

        # Energy Delivered Constraints
        constraints.update(self.energy_constraints(rates, active_sessions, self.period, self.enforce_energy_equality))

        # Peak Limit
        constraints.update(self.infrastructure_constraint(rates, infrastructure, constraint_type))

        # Objective Function
        objective = cp.Maximize(self.build_objective(rates, **kwargs))
        return {'objective': objective,
                'constraints': constraints,
                'variables': {'rates': rates}}



    def solve(self, active_sessions, prev_peak = 0, verbose: bool = False, **kwargs):
        """ Solve optimization problem to create a schedule of charging rates.

        Args:
            active_sessions (np.array[np.array(evse index, arriving time, duration, energy remaining)]): Two dimensional np.array (N * 4). 
                N represents all current & future EVs. Index of the second dimension are [0]: EVSE index, [1] : current time or arrival time
                [2] : job duration of charging job; [3] : current energy remaining.
            infrastructure (Dict[np.array, np.array, np.array, np.array): network infrastructure information, see class infrastructure_info in network.py
            constraint_type: currently support SOC and LINEAR
            verbose (bool): See cp.Problem.solve()

        Returns:
            np.Array: Numpy array of charging rates of shape (N, T) where N is the number of EVSEs in the network and
                T is the length of the optimization horizon. Rows are ordered according to the order of evse_index in
                infrastructure.
        """
        # Here we take in arguments which describe the problem and build a problem instance.
        if len(active_sessions) == 0:
            return np.zeros((self.max_ev, 1))
        problem_dict = self.build_problem(active_sessions, self.infrastructure, self.constraint_type, prev_peak, **kwargs)
        prob = cp.Problem(problem_dict['objective'], list(problem_dict['constraints'].values()))
        prob.solve(solver=self.solver, verbose=verbose)
        if prob.status not in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
            raise InfeasibilityException(f'Solve failed with status {prob.status}')
        return problem_dict['variables']['rates'].value