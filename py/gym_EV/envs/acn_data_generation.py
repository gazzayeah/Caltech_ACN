import math
from datetime import datetime
from datetime import timedelta
from acnportal.acndata import DataClient
import matplotlib.pyplot as plt
from gym_EV.envs.utils import get_moving_averages
import numpy as np
import pandas as pd
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



def get_solar_capacity(start: datetime, 
                    end: datetime, 
                    solarFilePath = './solar.csv', 
                    maxCapacity = 20, 
                    period = 0.1):
  """
  Obtain dictionary-based solar time-series, mapping timestamps (minutes) to power (Watts).
  Default filename of solar time-series is solar.csv.
  
  Args:
  solarFilePath (string): file name which solar data is retrieved from.
  maxCapacity (float): normalizing maximum of the solar time-series.
  
  Returns:
      solarCapacityDict (Dictionary(datetime: np.array)): instantaneous power provided by time-varying power source, eg. PV plant.
  """
  # read solar data as pandas dataframe
  df = pd.read_csv(solarFilePath)
  df["time"] = pd.to_datetime(df["time"])
  # start becomes to pandas timestamp starting from 5 am
  startTimestamp = pd.Timestamp(start + timedelta(hours = 5))
  # start becomes to pandas timestamp ending as 7 pm
  endTimestamp = pd.Timestamp(end - timedelta(hours = 5))
  if len(df.index[df['time'] == startTimestamp].tolist()) == 0:
    raise ValueError("Start Date : {0} is not in Database".format(startTimestamp))
  if len(df.index[df['time'] == endTimestamp].tolist()) == 0:
    raise ValueError("End Date {0} is not in Database".format(endTimestamp))
  tmpTimestamp = startTimestamp
  solarCapacityDict = {}
  while tmpTimestamp < end:
    dateIdx = datetime(tmpTimestamp.year, tmpTimestamp.month, tmpTimestamp.day)
    dailyCapacitySeries = []
    tmpdateIdx = dateIdx
    for delta in range(int(round(24 / period))):
      #print(pd.Timestamp(tmpdateIdx))
      searchList = df.index[df['time'] == pd.Timestamp(tmpdateIdx)].tolist()
      if len(searchList) == 0:
        dailyCapacitySeries.append(0)
      else:
        dailyCapacitySeries.append(df["power (W)"][searchList[0]])
      tmpdateIdx += timedelta(hours = period)
    # obtain moving average series by convolution
    dailyCapacitySeries = get_moving_averages(np.array(dailyCapacitySeries))
    # compute normalizing scaler
    scaler = maxCapacity / (np.max(dailyCapacitySeries) - np.min(dailyCapacitySeries))      
    solarCapacityDict[dateIdx] = dailyCapacitySeries * scaler
    # increment to next day 5 am
    tmpTimestamp += pd.Timedelta(days = 1)
  return solarCapacityDict
  


def generate_events(start: datetime, 
                    end: datetime, 
                    solarCapacityDict, 
                    token = "Js7k5LJ0qMUqESRv2PNHr2V4-09cD4A2tt6evDX5eIg", 
                    site = 'caltech', 
                    phase_selection = False, 
                    period = 0.1):
  """
  Return a list of EV charging sessions within the specified time range gathered from the acndata API.
  Args:
      token (str): API token needed to access the acndata API.
      site (str): ACN id for the site where data should be gathered.
      start (datetime): Only return sessions which began after start.
      end (datetime): Only return session which began before end.
      solarFilePath (string): file name which solar data is retrieved from.
      maxCapacity (float): normalizing maximum of the solar time-series.
      solarCapacityDict (Dictionary): instantaneous power provided by time-varying power source, eg. PV plant.
      phase_selection (bool): determine if to select phase type by ACN data or uniformly random.
  Returns:
      daily_evs (np.array(session * profile)): return all ev charging profiles in the current day.
  """
  evs = {1:np.array([])}
  tmpdate = start
  count = 1
  # initialize random seeds
  np.random.seed(0)
  while tmpdate <= end - timedelta(days = 1):
    if tmpdate in solarCapacityDict:
      variableCapacityArray = solarCapacityDict[tmpdate]
    else:
      variableCapacityArray = np.zeros(int(round(24 / period)))
    evs[count] = get_daily_events(tmpdate, tmpdate + timedelta(days = 1), token, site, variableCapacityArray, phase_selection, period)
    tmpdate = tmpdate + timedelta(days = 1)
    count += 1
  return evs


def get_daily_events(start, 
                     end, 
                     token, 
                     site, 
                     variableCapacityArray, 
                     phase_selection, 
                     period):
  """ 
  Return a list of daily EV charging sessions gathered from the acndata API.
  Args:
      token (str): API token needed to access the acndata API.
      site (str): ACN id for the site where data should be gathered.
      start (datetime): Only return sessions which began after start.
      end (datetime): Only return session which began before end.
      variableCapacityArray (np.array): daily instantaneous power provided by time-varying power source, eg. PV plant.
      phase_selection (bool): determine if to select phase type by ACN data or uniformly random.
  Returns:
      daily_evs (np.array(session * profile)): return all ev charging profiles in the current day.
  """
  client = DataClient(token)
  docs = client.get_sessions_by_time(site, start, end)
  daily_evs = np.array([])
  for d in docs:
    #print(d)
    daily_evs = np.append(daily_evs, _convert_to_ev_profile(d, variableCapacityArray, phase_selection, period))
  if daily_evs.size != 0:
    evs_matrix = np.reshape(daily_evs, (-1, 6))
    return evs_matrix[np.argsort(evs_matrix[:, 1])]
  else:
    return daily_evs


def _convert_to_ev_profile(d, 
                           variableCapacityArray, 
                           phase_selection,
                           period):
  """ Convert a json document for a single charging session from acndata into an EV object.
  Args:
      d (dict): Session expressed as a dictionary. See acndata API for more details.
      variableCapacityArray (np.array): daily instantaneous power provided by time-varying power source, eg. PV plant.
      phase_selection (bool): determine if to select phase type by ACN data or uniformly random.
  Returns:
      ev_profile (np.array([5])): EV profile with data from the acndata session doc with six elements. [1] phase selection, [2] arrival time
      [3] duration, [4] energy requested, [5] maximum power intake, [6] network capacity when ev arrives
  """
  # initialize ev profile vector
  ev_profile = np.array([])
  
  # check if data is valid
  if d["kWhDelivered"] == None or d["connectionTime"] == None or d["disconnectTime"] == None or d["spaceID"] == None:
    return ev_profile
  else:  
    arrival_datetime = d["connectionTime"]
    quant_arrival_timestep = np.floor((arrival_datetime.hour + arrival_datetime.minute / 60 + arrival_datetime.second / 3600) / period) * period
    #solar_timestep = arrival_datetime.replace(minute = int(round(arrival_datetime.minute / 6)) * 6, second = 0, microsecond = 0).timestamp()
    departure_datetime = d["disconnectTime"]
    next_datetime = arrival_datetime.replace(hour = 0, minute = 0, second = 0, microsecond = 0) + timedelta(days = 1)
    if departure_datetime <= next_datetime:
      quant_departure_timestep = np.ceil((departure_datetime.hour + departure_datetime.minute / 60 + departure_datetime.second / 3600) / period) * period
      energy_remaining = d["kWhDelivered"]     
    else:
      quant_departure_timestep = 24
      energy_remaining = d["kWhDelivered"] * ((next_datetime.timestamp() - arrival_datetime.timestamp()) / (departure_datetime.timestamp() - arrival_datetime.timestamp()))
      
    maxRate = 10 #abs(random.normal(0, 5))
    # get realtime capacity by quantized arrival time
    capacity = variableCapacityArray[int(round(quant_arrival_timestep / period))]

    ev_profile = np.append(ev_profile, [quant_arrival_timestep, quant_departure_timestep - quant_arrival_timestep, energy_remaining, maxRate, capacity])
    if phase_selection:
      # encode ev station id as [-1, 0, 1]
      for keys in station_id_caltech_dict:
        if d["spaceID"] in station_id_caltech_dict[keys]:
          ev_profile = np.append(keys, ev_profile)
          break
        else: continue  
    else:
      # choose phase line uniformly random
      ev_profile = np.append(random.randint(low = -1, high = 2), ev_profile)
  
  if ev_profile.size == 6:
    return ev_profile
  else:
    raise ValueError("Invalid EV profile output: {0}, should be in dimension of 6".format(ev_profile))
  
  


########################################################
#
# Run as the main module (eg. for testing).
#
########################################################  
if __name__ == "__main__":  
  from datetime import datetime
  def plot_TS(EVProfileMatrix, label = ['AB', 'BC', 'CA']): 
    """
    Plot time-series of EV profile and its one-step prediction from prediction matrix.
  
    Args:
    EVProfileMatrix (np.array(n_arrival * 6)): matrix containing all ev arrival profile.
  
    Return:
        predSeq (np.array(numData * EVProfile)): one-step predicted sequence.
        realSeq (np.array(numData * EVProfile)): original testing data sequence.
    """
  
    # get real and one-step predicted time-series ready to plot
    ABMatrix = EVProfileMatrix[np.where(EVProfileMatrix[:, 0] == -1)[0], 1 : 6]
    BCMatrix = EVProfileMatrix[np.where(EVProfileMatrix[:, 0] == 0)[0], 1 : 6]
    CAMatrix = EVProfileMatrix[np.where(EVProfileMatrix[:, 0] == 1)[0], 1 : 6]
    phaseMatrix = {0:ABMatrix, 1:BCMatrix, 2:CAMatrix}
    for i in range(3):
      fig, axs = plt.subplots(5)
      fig.suptitle('Time Series Analysis on {0} Phase'.format(label[i]))
      axs[0].plot(phaseMatrix[i][:, 0])
      axs[0].set_title('Arrival Time')
      axs[0].set(xlabel='EV Arrivals', ylabel='Arrival Time')
      axs[1].plot(phaseMatrix[i][:, 1])
      axs[1].set_title('Duration')
      axs[1].set(xlabel='EV Arrivals', ylabel='Duration')
      axs[2].plot(phaseMatrix[i][:, 2])
      axs[2].set_title('Energy Remaining')
      axs[2].set(xlabel='EV Arrivals', ylabel='Energy Remaining')
      axs[3].plot(phaseMatrix[i][:, 3])
      axs[3].set_title('Maximum Rate')
      axs[3].set(xlabel='EV Arrivals', ylabel='Maximum Rate')
      axs[4].plot(phaseMatrix[i][:, 4])
      axs[4].set_title('Solar Capacity')
      axs[4].set(xlabel='EV Arrivals', ylabel='Solar Capacity')    
      # Hide x labels and tick labels for top plots and y ticks for right plots.
      for ax in axs.flat:
        ax.label_outer()
      plt.show()


  start = datetime(2018, 5, 25)
  end = datetime(2018, 5, 28)
  #c = generate_events(start, end)
  solarCapacityDict = get_solar_capacity(start, end, solarFilePath = './solar.csv', maxCapacity = 30, period = 0.1)
  databyDate = generate_events(start, end, solarCapacityDict)
  dataWeekdays = {}
  c = 0
  for i in range(len(databyDate)):
    if (start + timedelta(days = i)).weekday()  <= 4:
      c += 1
      dataWeekdays[c] = databyDate[i + 1]
  dataMatrix = dataWeekdays[1]
  for i in range(2, len(dataWeekdays) + 1):
    dataMatrix = np.append(dataMatrix, dataWeekdays[i], axis = 0)
    
  plot_TS(dataMatrix)
    
  #y = np.array([])
  #for keys in x:
    #y = np.append(y, x[keys])
  #plt.figure("Solar Time Series")
  #plt.plot(y)
  #plt.show()
  # example output : c = {1: array([[-1.        ,  6.38833333, 12.50638889, 13.41      ,  3.        ,  15.08204433], ...]}