from typing import List, Union
from collections import namedtuple
import numpy as np
import cvxpy as cp

# Define object of objective function: includes function, temperature coefficient and kwargs that the function needs as the input
ObjectiveComponent = namedtuple('ObjectiveComponent', ['function', 'coefficient', 'kwargs'], defaults=[1, {}])

class AdaptiveChargingOptimization:
    """ Base class for all MPC based charging algorithms.

    Args:
        objective (List[ObjectiveComponent]): List of components which make up the optimization objective.
        max_ev: maximum EV network can take (number of EVSEs)
        max_rate: maximum rate an EVSE can assign
        max_capacity: peak rate the network can deliver
        period: time interval that new network data is retrieved
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
            active_sessions (List[SessionInfo]): List of SessionInfo objects for all active charging sessions.
            max_rate: maximum rate an EVSE can assign
            period: time interval that new network data is retrieved

        Returns:
            List[cp.Constraint]: List of lower bound constraint, upper bound constraint.
        """
        lb, ub = np.zeros(rates.shape), np.zeros(rates.shape)
        activeNum = len(active_sessions['index'])
        for i in range(activeNum):
            idx = int(active_sessions["index"][i])
            # for ev i, compute quantized remaining time
            qtz_duration = int(active_sessions["remain_time"][i] // period) + 1
            lb[idx, 0 : qtz_duration] = 0
            ub[idx, 0 : qtz_duration] = max_rate          
            
        # To ensure feasibility, replace upper bound with lower bound when they conflict
        ub[ub < lb] = lb[ub < lb]
        return {'charging_rate_bounds.lb': rates >= lb, 'charging_rate_bounds.ub': rates <= ub}

    @staticmethod
    def energy_constraints(rates: cp.Variable, active_sessions, period, enforce_energy_equality=False):
        """ Get constraints on the energy delivered for each session.

        Args:
            rates (cp.Variable): cvxpy variable representing all charging rates. Shape should be (N, T) where N is the
                total number of EVSEs in the system and T is the length of the optimization horizon.
            active_sessions (List[SessionInfo]): List of SessionInfo objects for all active charging sessions.
            max_rate (float): maximum rate an EVSE can assign
            period (float): time interval that new network data is retrieved
            enforce_energy_equality (bool): If True, energy delivered must be equal to energy requested for each EV.
                If False, energy delivered must be less than or equal to request.

        Returns:
            List[cp.Constraint]: List of energy delivered constraints for each session.
        """
        constraints = {}
        activeNum = len(active_sessions['index'])
        for i in range(activeNum):
            idx = int(active_sessions["index"][i])
            # for ev i, compute quantized remaining time
            qtz_duration = int(active_sessions["remain_time"][i] // period) + 1            
            planned_energy = cp.sum(rates[idx, 0 : qtz_duration])
            planned_energy *= period
            constraint_name = f'energy_constraints.{idx}'
            if enforce_energy_equality:
                constraints[constraint_name] = planned_energy == active_sessions["remain_energy"][i]
            else:
                constraints[constraint_name] = planned_energy <= active_sessions["remain_energy"][i]
        return constraints

    @staticmethod
    def infrastructure_constraint(rates: cp.Variable, infrastructure, constraint_type):
        """ Get constraints enforcing infrastructure limits.

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
        obj = cp.Constant(0)
        for component in self.objective_configuration:
            obj += component.coefficient * component.function(rates)
        return obj

    def build_problem(self, active_sessions, infrastructure, constraint_type, prev_peak:float=0):
        """ Build parts of the optimization problem including variables, constraints, and objective function.

        Args:
            active_sessions (List[SessionInfo]): List of SessionInfo objects for all active charging sessions.
            infrastructure (Dict[np.array, np.array, np.array, np.array): network infrastructure information, see class infrastructure_info in network.py
            constraint_type: currently support SOC and LINEAR
            prev_peak (float): Previous peak current draw during the current billing period.

        Returns:
            Dict[str: object]:
                'objective' : cvxpy expression for the objective of the optimization problem
                'constraints': list of all constraints for the optimization problem
                'variables': dict mapping variable name to cvxpy Variable.
        """
        optimization_horizon = max(int(s // self.period) + 1 for s in active_sessions['remain_time'])
        rates = cp.Variable(shape=(self.max_ev, optimization_horizon))
        constraints = {}

        # Rate constraints
        constraints.update(self.charging_rate_bounds(rates, active_sessions, self.max_rate, self.period))

        # Energy Delivered Constraints
        constraints.update(self.energy_constraints(rates, active_sessions, self.period, self.enforce_energy_equality))

        # Peak Limit
        constraints.update(self.infrastructure_constraint(rates, infrastructure, constraint_type))

        # Objective Function
        objective = cp.Maximize(self.build_objective(rates, prev_peak=prev_peak))
        return {'objective': objective,
                'constraints': constraints,
                'variables': {'rates': rates}}

    def solve(self, active_sessions, prev_peak = 0, verbose: bool = False):
        """ Solve optimization problem to create a schedule of charging rates.

        Args:
            active_sessions (List[SessionInfo]): List of SessionInfo objects for all active charging sessions.
            infrastructure (Dict[np.array, np.array, np.array, np.array): network infrastructure information, see class infrastructure_info in network.py
            constraint_type: currently support SOC and LINEAR
            verbose (bool): See cp.Problem.solve()

        Returns:
            np.Array: Numpy array of charging rates of shape (N, T) where N is the number of EVSEs in the network and
                T is the length of the optimization horizon. Rows are ordered according to the order of evse_index in
                infrastructure.
        """
        # Here we take in arguments which describe the problem and build a problem instance.
        if len(active_sessions['index']) == 0:
            return np.zeros((self.max_ev, 1))
        problem_dict = self.build_problem(active_sessions, self.infrastructure, self.constraint_type, prev_peak)
        prob = cp.Problem(problem_dict['objective'], list(problem_dict['constraints'].values()))
        prob.solve(solver=self.solver, verbose=verbose)
        if prob.status not in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
            raise InfeasibilityException(f'Solve failed with status {prob.status}')
        return problem_dict['variables']['rates'].value


# Objective Functions

"""
Inputs: 
    rates -> cp.Variable()
    
Sometime needed
    infrastructure 
    active_sessions
    period
    
Signals -> signals_interface?
    previous_rates
    energy_tariffs 
    demand_charge
    external_demand
    signal_to_follow
"""

# ---------------------------------------------------------------------------------
#  Objective Functions
#
#
#  All objectives should take rates as their first positional argument.
#  All other arguments should be passed as keyword arguments.
#  All functions should except **kwargs as their last argument to avoid errors
#  when unknown arguments are passed.
#
# ---------------------------------------------------------------------------------


def quick_charge(rates):
    """
    Objective function that encourages network to charge as fast as possible
    
    Args:
        Charging-rate matrix.
    
    Return:
       Output of objective function.
    """
    optimization_horizon = rates.shape[1]
    c = np.array([(optimization_horizon - t) / optimization_horizon for t in range(optimization_horizon)])
    return c @ cp.sum(rates, axis=0)

'''
def charging_power(rates, infrastructure, **kwargs):
    """ Returns a matrix with the same shape as rates but with units kW instead of A. """
    voltage_matrix = np.tile(infrastructure.voltages, (rates.shape[1], 1)).T
    return cp.multiply(rates, voltage_matrix) / 1e3


def aggregate_power(rates, infrastructure, **kwargs):
    """ Returns aggregate charging power for each time period. """
    return cp.sum(charging_power(rates, infrastructure=infrastructure), axis=0)


def get_period_energy(rates, infrastructure, period, **kwargs):
    """ Return energy delivered in kWh during each time period and each session. """
    power = charging_power(rates, infrastructure=infrastructure)
    period_in_hours = period / 60
    return power * period_in_hours


def aggregate_period_energy(rates, infrastructure, interface, **kwargs):
    """ Returns the aggregate energy delivered in kWh during each time period. """
    # get charging rates in kWh per period
    energy_per_period = get_period_energy(rates, infrastructure=infrastructure, period=interface.period)
    return cp.sum(energy_per_period, axis=0)


def equal_share(rates, infrastructure, interface, **kwargs):
    return -cp.sum_squares(rates)


def tou_energy_cost(rates, infrastructure, interface, **kwargs):
    current_prices = interface.get_prices(rates.shape[1])    # $/kWh
    return -current_prices @ aggregate_period_energy(rates, infrastructure, interface)


def total_energy(rates, infrastructure, interface, **kwargs):
    return cp.sum(get_period_energy(rates, infrastructure, interface.period))


def peak(rates, infrastructure, interface, baseline_peak=0, **kwargs):
    agg_power = aggregate_power(rates, infrastructure)
    max_power = cp.max(agg_power)
    prev_peak = interface.get_prev_peak() * infrastructure.voltages[0] / 1000
    if baseline_peak > 0:
        return cp.maximum(max_power, baseline_peak, prev_peak)
    else:
        return cp.maximum(max_power, prev_peak)


def demand_charge(rates, infrastructure, interface, baseline_peak=0, **kwargs):
    p = peak(rates, infrastructure, interface, baseline_peak, **kwargs)
    dc = interface.get_demand_charge()
    return -dc * p


def load_flattening(rates, infrastructure, interface, external_signal=None, **kwargs):
    if external_signal is None:
        external_signal = np.zeros(rates.shape[1])
    aggregate_rates_kW = aggregate_power(rates, infrastructure)
    total_aggregate = aggregate_rates_kW + external_signal
    return -cp.sum_squares(total_aggregate)
'''

# def smoothing(rates, active_sessions, infrastructure, previous_rates, normp=1, *args, **kwargs):
#     reg = -cp.norm(cp.diff(rates, axis=1), p=normp)
#     prev_mask = np.logical_not(np.isnan(previous_rates))
#     if np.any(prev_mask):
#         reg -= cp.norm(rates[0, prev_mask] - previous_rates[prev_mask], p=normp)
#     return reg

