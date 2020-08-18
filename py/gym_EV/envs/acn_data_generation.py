import math
from datetime import datetime
from datetime import timedelta
from acnportal.acndata import DataClient
import numpy as np
from numpy import random


# Caltech Phase-line Construction
# Define the sets of EVSEs in the Caltech ACN.
CC_pod_ids = [
    "CA-322",
      "CA-493",
      "CA-496",
      "CA-320",
      "CA-495",
      "CA-321",
      "CA-323",
      "CA-494",
  ]
AV_pod_ids = [
    "CA-324",
      "CA-325",
      "CA-326",
      "CA-327",
      "CA-489",
      "CA-490",
      "CA-491",
      "CA-492",
  ]
AB_ids = (
    ["CA-{0}".format(i) for i in [308, 508, 303, 513, 310, 506, 316, 500, 318, 498]]
      + AV_pod_ids
      + CC_pod_ids
  )
BC_ids = [
    "CA-{0}".format(i)
      for i in [304, 512, 305, 511, 313, 503, 311, 505, 317, 499, 148, 149, 212, 213]
  ]
CA_ids = [
    "CA-{0}".format(i)
      for i in [307, 509, 309, 507, 306, 510, 315, 501, 319, 497, 312, 504, 314, 502]
  ]

# represent AB line as -1, BC as 0 and CA as 1
station_id_caltech_dict = {-1: AB_ids, 0: BC_ids, 1: CA_ids}



def generate_events(start: datetime, end: datetime, token = "Js7k5LJ0qMUqESRv2PNHr2V4-09cD4A2tt6evDX5eIg", site = 'caltech', phase_selection=False):
  """ Return EventQueue filled using events gathered from the acndata API.
  Args:
      See get_evs().
  Returns:
      EventQueue: An EventQueue filled with Events gathered through the acndata API.
  """
  evs = {1:np.array([])}
  tmpdate = start
  count = 1
  while tmpdate <= end - timedelta(days = 1):
    evs[count] = get_daily_events(tmpdate, tmpdate + timedelta(days = 1), token, site, phase_selection)
    tmpdate = tmpdate + timedelta(days = 1)
    count += 1
  return evs


def get_daily_events(start, end, token, site, phase_selection):
  """ Return a list of EVs gathered from the acndata API.
  Args:
      token (str): API token needed to access the acndata API.
      site (str): ACN id for the site where data should be gathered.
      start (datetime): Only return sessions which began after start.
      end (datetime): Only return session which began before end.
      period (int): Length of each time interval. (minutes)
      voltage (float): Voltage of the network.
      max_battery_power (float): Default maximum charging power for batteries.
      max_len (int): Maximum length of a session. (periods) Default None.
      battery_params (Dict[str, object]): Dictionary containing parameters for the EV's battery. Three keys are
          supported. If none, Battery type is used with default configuration. Default None.
          - 'type' maps to a Battery-like class. (required)
          - 'capacity_fn' maps to a function which takes in the the energy delivered to the car, the length of the
              session, the period of the simulation, and the voltage of the system. It should return a tuple with
              the capacity of the battery and the initial charge of the battery both in A*periods.
          - 'kwargs' maps to a dictionary of keyword arguments which will be based to the Battery constructor.
      force_feasible (bool): If True, the requested_energy of each session will be reduced if it exceeds the amount
          of energy which could be delivered at maximum rate during the duration of the charging session.
          Default False.
  Returns:
  """
  client = DataClient(token)
  docs = client.get_sessions_by_time(site, start, end)
  daily_evs = np.array([])
  for d in docs:
    daily_evs = np.append(daily_evs, _convert_to_ev_profile(d, phase_selection))
  if daily_evs != np.array([]):
    evs_matrix = np.reshape(daily_evs, (-1, 4))
    return evs_matrix[np.argsort(evs_matrix[:, 0])]
  else:
    return daily_evs


def _convert_to_ev_profile(d, phase_selection):
  """ Convert a json document for a single charging session from acndata into an EV object.
  Args:
      d (dict): Session expressed as a dictionary. See acndata API for more details.
      offset (int): Simulation timestamp of the beginning of the simulation.
      See get_evs() for additional args.
  Returns:
      EV: EV object with data from the acndata session doc.
  """
  # initialize ev profile vector
  ev_profile = np.array([])
  
  # check if data is valid
  if d["kWhDelivered"] == None or d["connectionTime"] == None or d["disconnectTime"] == None or d["spaceID"] == None:
    return ev_profile
  else:  
    arrival_datetime = d["connectionTime"]
    departure_datetime = d["disconnectTime"]
    next_datetime = arrival_datetime.replace(hour = 0, minute = 0, second = 0, microsecond = 0) + timedelta(days = 1)
    if departure_datetime <= next_datetime:
      job_deadline = departure_datetime.timestamp() - arrival_datetime.timestamp()
      energy_remaining = d["kWhDelivered"]
      arrival_time = arrival_datetime.hour + arrival_datetime.minute / 60 + arrival_datetime.second / 3600
    else:
      job_deadline = next_datetime.timestamp() - arrival_datetime.timestamp()
      energy_remaining = d["kWhDelivered"] * (job_deadline / (departure_datetime.timestamp() - arrival_datetime.timestamp()))
      arrival_time = arrival_datetime.hour + arrival_datetime.minute / 60 + arrival_datetime.second / 3600  
    ev_profile = np.append(ev_profile, [arrival_time, job_deadline / 3600, energy_remaining])
    
    if phase_selection:
      # encode ev station id as [-1, 0, 1]
      for keys in station_id_caltech_dict:
        if d["spaceID"] in station_id_caltech_dict[keys]:
          ev_profile = np.append(ev_profile, keys)
          break
        else: continue  
    else:
      # choose phase line uniformly random
      ev_profile = np.append(ev_profile, random.randint(low = -1, high = 2))
  
  if ev_profile.size == 4:
    return ev_profile
  else:
    raise ValueError("Invalid EV profile output: {0}, should be in dimension of 4".format(ev_profile))
