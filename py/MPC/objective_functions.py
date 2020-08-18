from typing import List, Union
from collections import namedtuple
import numpy as np
import cvxpy as cp



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
# Define object of objective function: includes function, temperature coefficient and kwargs that the function needs as the input
ObjectiveComponent = namedtuple('ObjectiveComponent', ['function', 'coefficient', 'kwargs'], defaults=[1, {}])

def quick_charge(rates, **kwargs):
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


def laxity_first(rates, **kwargs):
    active_sessions = kwargs['session']
    max_rate = kwargs['MAX_RATE']
    return cp.sum(active_sessions['remain_time'] - (active_sessions['remain_energy'] - rates[active_sessions['index'].astype(int), 0] * 0.1) / max_rate)

def l1_aggregate_power(rates, **kwargs):
    """ Returns L2 norm of aggregate charging power for each time period. """
    #cp.norm(rates, "fro")
    return cp.sum(cp.sum(rates, axis = 1))


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


# def smoothing(rates, active_sessions, infrastructure, previous_rates, normp=1, *args, **kwargs):
#     reg = -cp.norm(cp.diff(rates, axis=1), p=normp)
#     prev_mask = np.logical_not(np.isnan(previous_rates))
#     if np.any(prev_mask):
#         reg -= cp.norm(rates[0, prev_mask] - previous_rates[prev_mask], p=normp)
#     return reg

