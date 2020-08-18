from datetime import timedelta
from datetime import datetime
from gym_EV.envs import acn_data_generation

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler

import torch
import torch.nn as nn
import torch.nn.functional as F

from lstm_hyper import *
from utils import *


def get_acn_data(start : datetime, 
                 end : datetime, 
                 phase_selection : bool = True):
    """
    Obtain ACN data within a time range. Only weekday data will be extracted.

    Args:
    start (datetime object): start date of ACN data.
    end (datetime object): end date of ACN data.
    phase_selection (Bool): follow real phase selection in ACN if True, or unirformly choosing phase if False.

    Return:
    data_list(List[data]): ACN data with ascending arrival data with ascending arrival time. No daily partition.
    """
    # iterator through each date
    iter_date = start
    # concatenated data seuqences
    data_list = np.array([[]])
    while iter_date <= end:
        if  iter_date.weekday() < 5:
            data = acn_data_generation.generate_events(iter_date, iter_date + timedelta(days = 1), phase_selection = phase_selection)
            if data[1].size > 0:
                # append data as a whole
                if data_list.size == 0:
                    data_list = data[1]
                else:
                    data_list = np.append(data_list, data[1], axis = 0)
        iter_date += timedelta(days = 1)
    return data_list



class LSTM_DATA():
    """
    Convert ACN EV time series into LSTM input-output data form. 
    Input is the sequence of historical data of EV profile: s_1, ..., s_T; 
    ouput is the next occuring EV profile in network/phase: s_T+1, ..., s_T+n.
    One I/O can form a supervised-learning data and the conversion can be phase-wise partitioned.
    
    Args:
        start (datetime object): start date of ACN data.
        end (datetime object): end date of ACN data.
        phase_selection (Bool): follow real phase selection in ACN if True, or unirformly choosing phase if False.
    """
    def __init__(self, 
                 start : datetime, 
                 endTrain : datetime, 
                 endTest : datetime, 
                 windowSize : int = 100, 
                 phaseSelection : bool = True):
        # obtain training data matrix
        self.dataTrain = get_acn_data(start, endTrain, phase_selection = phaseSelection)
        self.phaseTrain = self.dataTrain[:, 3]
        self.dataTrain = self.dataTrain[:, 0 : 3]
        # obtain testing data matrix
        self.dataTest = get_acn_data(endTrain, endTest, phase_selection = phaseSelection)
        self.phaseTest = self.dataTest[:, 3]
        self.dataTest = self.dataTest[:, 0 : 3]        
        # initialize min-max normalization scaler between (-1, 1), based on training data
        self.scaler = MinMaxScaler(feature_range=(-1, 1))
        # normalize training data
        self.dataTrainNormed = self.scaler.fit_transform(self.dataTrain.reshape(len(self.dataTrain), -1))
        # normalize testing data by training data normalizer
        self.dataTestNormed = self.scaler.transform(self.dataTest.reshape(len(self.dataTest), -1))
        if (windowSize > len(self.dataTrain) - 1):
            raise ValueError("LSTM Memory Size is Too Large : {0} > training data length {1}".format(windowSize, len(self.dataTrain)))
        else:
            # memory size of LSTM
            self.windowSize = windowSize
        # create phase data dict where types -1 = AB, 0 = BC, 1 = CA.
        '''
        self.phase_data = { -1 : d.dataTest[np.where(d.phaseTest[:] == -1)[0],0:3],
                            0 : self.data_list[np.where(self.data_list[:, 3] == 0)[0],0:3],
                            1 : self.data_list[np.where(self.data_list[:, 3] == 1)[0],0:3]}
        '''
        
        
        
    def get_normed_sequence_byphase(self, numVariables : int = 3, 
                                    phaseType = None):
        """
        Obtain ACN data in supervised-learning form to feed in the lstm learning agent.
        
        Args:
        numVariabels (int) : decide how many variables in EV proflie are included in the sequence
        phaseType (Union(None, int)): type of phase line: -1 = AB, 0 = BC, 1 = CA, None = All.
        
        Return:
        Dict{"train" : torch.FloatTensor, "test" : torch.FloatTensor} : a dictionary containing both training sequence and testing sequence
        """        
        if phaseType == None:
            DataTrain = self.dataTrainNormed[:, 0:numVariables]
            DataTest = self.dataTestNormed[:, 0:numVariables]     
        elif phaseType in [-1, 0, 1]:
            DataTrain = self.dataTrainNormed[np.where(self.phaseTrain == phaseType)[0],0:numVariables]
            DataTest = self.dataTestNormed[np.where(self.phaseTest == phaseType)[0],0:numVariables]
        else: 
            raise ValueError("Invalid phase type: should be in [None, -1, 0, 1] but {0}".format(phaseType))
        trainInoutSequence = self.create_inout_sequences(torch.FloatTensor(DataTrain), self.windowSize)  
        testInoutSequence =  self.create_inout_sequences(torch.FloatTensor(np.append(DataTrain[-self.windowSize:], DataTest, axis = 0)), self.windowSize) 
        return {"train": trainInoutSequence, "test": testInoutSequence}
    
    
    
    def create_inout_sequences(self, input_data, 
                               tw):
        '''
        Convert time series into (data : label) based data sequences.
        
        Args:
        input_data (np.array[np.array]): input data of lstm learning agents.
        tw (int): length of training data
        
        Return:
        inout_seq(List[tuple(np.array, np.array)]): ouput will be a List of tuple containing each input data with its output label.
        '''
        inout_seq = []
        L = len(input_data)
        for i in range(L-tw):
            train_seq = input_data[i:i+tw]
            train_label = input_data[i+tw:i+tw+1]
            inout_seq.append((train_seq ,train_label))
        return inout_seq
    


class LSTM(nn.Module):
    """
    """
    def __init__(self, input_size=1, hidden_layer_size=50, output_size=1):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size
        self.hidden_layer_size_1 = hidden_layer_size * 2
    
        # inputs are the elements of sequence {12}: passengers of each month
        self.lstm = nn.LSTM(input_size, hidden_layer_size)
    
        self.linear1 = nn.Linear(hidden_layer_size, self.hidden_layer_size_1)
    
        self.linear2 = nn.Linear(self.hidden_layer_size_1, output_size)
    
        self.hidden_cell = (torch.zeros(1,1,self.hidden_layer_size),
                                torch.zeros(1,1,self.hidden_layer_size))
    
    def forward(self, input_seq):
        """
        """
        #print(input_seq.view(len(input_seq), 1, -1))
        lstm_out, self.hidden_cell = self.lstm(input_seq.view(len(input_seq), 1, -1), self.hidden_cell)
        #print(lstm_out.size())
        predictions = F.relu(self.linear1(lstm_out.view(len(input_seq), -1)))
        #print(predictions.size())
        predictions = self.linear2(predictions.view(len(input_seq), -1))
        #print(pridictions.size())
        return predictions[-1].view(1, -1)
    
    
    
def train_lstm(trainInoutSequence, 
               model, optimizer, 
               loss_function = nn.MSELoss(), 
               epochs : int = 500, 
               PATH = "./runs/model/LSTM_AB.pt"):
    """
    No-batch LSTM training with same length (windowSize) of input sequence. Autosave of trained model's parameters to default directory.
    
    Args:
    trainInoutSequence (List[tuple(np.array, np.array)]): Inout sequences with supervised-learning labels for training.
    model (nn.module): lstm model with learnable parameters.
    optimizer (torch.optim): optimizer for updating model's parameters.
    loss_function (nn.module): loss function for gradient descent, default as MSE.
    epochs (int): epochs for LSTM repeated training.
    PATH (str): file path and name to save the learned model.
    
    Return:
    model (nn.module): lstm model with learned parameters.
    optimizer (torch.optim): optimizer with updated states.
    """
    model.train()
    for i in range(epochs):
        for seq, labels in trainInoutSequence:
            #print(seq)
            optimizer.zero_grad()
            model.hidden_cell = (torch.zeros(1, 1, model.hidden_layer_size),
                                 torch.zeros(1, 1, model.hidden_layer_size))
    
            y_pred = model(seq)
            #print((y_pred, labels))
            
            single_loss = loss_function(y_pred, labels)
            #print(single_loss)
            single_loss.backward()
            optimizer.step()
    
        if i%25 == 1:
            print(f'epoch: {i:3} loss: {single_loss.item():10.8f}')
            save_model(model, optimizer, PATH = PATH)
    print(f'epoch: {i:3} loss: {single_loss.item():10.10f}') 
    save_model(model, optimizer, PATH = PATH)
    print("Training Finished.")
    model.eval()
    return model, optimizer



def nStep_prediction(model, 
                          scaler, 
                          testSequence, 
                          windowSize : int = 30,  
                          nStep : int = 1):
    """
    Based on one input sequence with specified windowSize same as model sequence dimension, predict the next n incoming 
    data.
    
    Args:
    model (nn.module): lstm model with learned parameters.
    scaler (MinMaxScaler): scaler based on the training data to convert normalized testing data to original scale.
    testSequence (List[np.array]): Input sequence for testing.
    windowSize (int): memory size of LSTM.
    nStep (int): number of prediction made for the next n incoming data.
    
    Return:
    actual_predictions (np.array): np.array contains n-step predictions
    """
    # convert test_inputs to np.array
    test_inputs = testSequence[-windowSize:]
    if nStep == 0:
        return np.array([])
    for i in range(nStep):
        seq = torch.FloatTensor(test_inputs)
        with torch.no_grad():
            model.hidden = (torch.zeros(1, 1, model.hidden_layer_size),
                            torch.zeros(1, 1, model.hidden_layer_size))
            #test_inputs = np.append(test_inputs, np.array(model(seq).tolist()), axis = 0)
            test_inputs = np.append(test_inputs, model(seq).numpy(), axis = 0)
    # n step predictions are included in actural_prediction variable
    actual_predictions = scaler.inverse_transform(test_inputs[-nStep:])
    return actual_predictions




def merge_prediction_to_active_sessions(actualPredictions, activeSessions, currentTime, phaseEVSE):
    """
    Cleanse raw data from LSTM output into applicable format to the ACN network. All active sessions must be in the same phase type.
        The output should be a two dimensional np.array (N * 4). N represents all future EVs. 
    
    Args:
    actualPrediction (np.array): np.array contains n-step predictions;
    activeSessions (np.array[np.array(evse index, arriving time, duration, energy remaining)]): Two dimensional np.array (N * 4). 
                N represents all current & future EVs. Index of the second dimension are [0]: EVSE index, [1] : current time or arrival time
                [2] : job duration of charging job; [3] : current energy remaining;
    currentTime (float): current time step (from EV_env);
    phaseEVSE (List) : list of all EVSE IDs with the same phaseType;
        
    Return:
    futureSessions (np.array[np.array(evse index, arriving time, duration, energy remaining)]): Two dimensional np.array (N * 4). 
                N represents all current & future EVs. Index of the second dimension are [0]: EVSE index, [1] : current time or arrival time
                [2] : job duration of charging job; [3] : current energy remaining.
    """
    # localize the variable activeSessions
    activeSessions = activeSessions
    
    ########################################################
    #
    # Filter and rectify output of LSTM module
    #
    ######################################################## 
    # initialize output vectors
    futureSessions = []
    for event in actualPredictions:
        # if predicted arrival time is smaller than present or exceeds 24 hours
        if event[0] < currentTime or event[0]  > 24 or event[1] <=0 or event[2] <= 0:
            continue
        # rectify arrival time by current time offset
        event[0] -= currentTime
        # if it not the first predicting data
        if futureSessions != []:
            # arrival time (1st element of event) must be strictly increasing
            if (event[0] < futureSessions[-1][0]):
                continue
        # rectify duration and energy remaining by episode and arriving time
        if event[0 : 2].sum() >24:
            # shrink energy request by rectified duration
            event[2] = event[2] *  (24 - event[0]) / (event[1])
            # rectify duration
            event[1] = 24 - event[0]
        futureSessions.append(event)
        
    ########################################################
    #
    # Append the rectified sessions into active session list
    #
    ########################################################   
    for event in futureSessions:
        # no active session presented
        if activeSessions.size == 0:
            # randomly select over all available EVSE for the event and insert to the 1st position of the event array
            idx = np.random.choice([int(i) for i in phaseEVSE])
            # add event to activeSessions
            activeSessions = np.array([np.append(np.array([[idx]]), np.array([event]))])           
        else:
            # find current EVSEs that have jobs registered
            occupiedEVSE = set(activeSessions[:, 0])
            # find current EVSEs that have no job registered
            availableEVSE = set(phaseEVSE) - occupiedEVSE
            # iterate over all registered EVSEs
            for idx in occupiedEVSE:
                # extract the maximum job deadline for all session in EVSE idx and compare it with the arrival time of the event
                if event[0] >= max(activeSessions[np.where(activeSessions[:, 0] == idx)[0], 1 : 3].sum(axis = 1)):
                    # if the event arrives after a registered session
                    availableEVSE.add(idx)     
            # there is no available EVSE for this session
            if availableEVSE == set():
                continue
            else:
                # randomly select over all available EVSE for the event and insert to the 1st position of the event array
                idx = np.random.choice([int(i) for i in availableEVSE])
                event = np.append(np.array([[idx]]), np.array([event]))
                # add event to activeSessions
                activeSessions = np.append(activeSessions, np.array([event]), axis = 0)
    # sort by EVSE index to have better visualization  
    if activeSessions.size != 0:
        activeSessions = activeSessions[np.argsort(activeSessions[:, 0])]
    return activeSessions
            
            
        
                
                      
def get_optimizing_sessions(env, lstmData, model, phaseType = None):
    """
    Combine nStep_prediction and merge_prediction_to_active_sessions to obtain all sessions considered in the optimization process.
        This function requires information of EV_env and lstmData class.
    
    Args:
    env : gym environment of EV charging network;
    lstmData : preprocessing class converting ACN raw data to lstm data;
    model : LSTM learning model used for prediction purpose;
    phaseType (Union(None, int)): type of phase line: -1 = AB, 0 = BC, 1 = CA, None = All.
    
    Return:
    activeSessions (np.array[np.array(evse index, arriving time, duration, energy remaining)]): Two dimensional np.array (N * 4). 
                N represents all current & future EVs. Index of the second dimension are [0]: EVSE index, [1] : current time or arrival time
                [2] : job duration of charging job; [3] : current energy remaining;
    """
    chargingSession = env.get_previous_charging_events(phaseType = phaseType)[-lstmArgs.windowSize : ]
    normalizedSession = lstmData.scaler.transform(chargingSession.reshape(len(chargingSession), -1))
    actualPredictions = nStep_prediction(model, 
                                             lstmData.scaler, 
                                             testSequence = normalizedSession, 
                                             windowSize = lstmArgs.windowSize,  
                                             nStep = lstmArgs.nStep)        
    activeSessions = merge_prediction_to_active_sessions(actualPredictions, 
                                                         env.get_current_active_sessions(phaseType = phaseType), 
                                                         env.time, 
                                                         env.get_evse_id_by_phase(phaseType = phaseType))   
    return activeSessions        
    
    
    
def get_daily_offline_sessions_by_phase(data, phaseEVSE, phaseSelection = False, phaseType = None):
    """
    Obtain EV data with four types of information:  idx, 
    
    Args:
    data (np.array) : EV data of current date from ev_gym;
    phaseEVSE (List) : list of all EVSE IDs with the same phaseType;
    phase_selection (Bool): follow real phase selection in ACN if True, or unirformly choosing phase if False.
    phaseType (Union(None, int)): type of phase line: -1 = AB, 0 = BC, 1 = CA, None = All.
    
    Return:
    activeSessions (np.array[np.array(evse index, arriving time, duration, energy remaining)]): Two dimensional np.array (N * 4). 
                N represents all current & future EVs. Index of the second dimension are [0]: EVSE index, [1] : current time or arrival time
                [2] : job duration of charging job; [3] : current energy remaining;
    """    
    # sort data with ascending arrival sequence
    data = data[np.argsort(data[:, 0])]
    # convert phase ID list to set
    phaseSet = set(phaseEVSE)
    ########################################################
    #
    # Extract part from data that has the same phaseType
    #
    ########################################################       
    if phaseType == None:
        data  = data[:, 0 : 3]
    elif phaseType in [-1, 0, 1]:
        # filter all data with phaseType, outpust follows : [arrival duration, energy] ~ phaseType
        data  = data[np.where(data[:, 3] == phaseType)[0] , 0 : 3]
    else: 
        raise ValueError("Invalid phase type: should be in [None, -1, 0, 1] but {0}".format(phaseType))    
    ########################################################
    #
    # Append the rectified sessions into active session list
    #
    ########################################################       
    # initialize sessions
    activeSessions = np.array([])
    for event in data:     
        # no active sessions registered
        if activeSessions.size == 0:
            # randomly select over all available EVSE for the event and insert to the 1st position of the event array
            idx = np.random.choice([int(i) for i in phaseEVSE])
            event = np.append(np.array([[idx]]), np.array([event]))
            # add event to activeSessions
            activeSessions = np.array([event])       
        else:
            # find current EVSEs that have jobs registered
            occupiedEVSE = set(activeSessions[:, 0])
            # find current EVSEs that have no job registered
            availableEVSE = set(phaseEVSE) - occupiedEVSE  
            # iterate over all registered EVSEs
            for idx in occupiedEVSE:
                # extract the maximum job deadline for all session in EVSE idx and compare it with the arrival time of the event
                if event[0] >= max(activeSessions[np.where(activeSessions[:, 0] == idx)[0], 1 : 3].sum(axis = 1)):
                    # if the event arrives after a registered session
                    availableEVSE.add(idx)     
            # there is no available EVSE for this session
            if availableEVSE == set():
                print("WARNING (from get_daily_offline_sessions): new EV arrival is discarded due to deficient network feasibility. Discarded EV : {0}".format(event))
                continue
            else:
                # randomly select over all available EVSE for the event and insert to the 1st position of the event array
                idx = np.random.choice([int(i) for i in availableEVSE])
                event = np.append(np.array([[idx]]), np.array([event]))
                # add event to activeSessions
                activeSessions = np.append(activeSessions, np.array([event]), axis = 0)            
    # sort by EVSE index to have better visualization  
    return activeSessions[np.argsort(activeSessions[:, 0])]    
    
    
    



def get_daily_offline_sessions(data):
    """ Combine all active sessions in different phase line by get_daily_offline_sessions_by_phase"""
    sessionAB = get_daily_offline_sessions_by_phase(data, 
                               [i for i in range(0, netArgs.PHASE_PARTITION[0])], 
                               phaseSelection = dataArgs.PHASE_SELECT, 
                               phaseType = -1)
    sessionBC = get_daily_offline_sessions_by_phase(data, 
                               [i for i in range(netArgs.PHASE_PARTITION[0], sum(netArgs.PHASE_PARTITION))], 
                               phaseSelection = dataArgs.PHASE_SELECT, 
                               phaseType = 0)
    sessionCA = get_daily_offline_sessions_by_phase(data, 
                               [i for i in range(sum(netArgs.PHASE_PARTITION), gymArgs.MAX_EV)], 
                               phaseSelection = dataArgs.PHASE_SELECT, 
                               phaseType = 1)
    return np.concatenate((sessionAB, sessionBC, sessionCA), axis = 0)
        













########################################################
#
# Run as the main module (eg. for testing).
#
########################################################  
if __name__ == "__main__":
    '''
    Preprocess data and initialize hyperparameters
    '''
    # construct LSTM model, optimizer. loss function is MSE as default
    model_AB = LSTM(input_size=lstmArgs.numVariables, hidden_layer_size=50, output_size=lstmArgs.numVariables)
    optim_AB = torch.optim.Adam(model_AB.parameters(), lr=0.001)    
    model_BC = LSTM(input_size=lstmArgs.numVariables, hidden_layer_size=50, output_size=lstmArgs.numVariables)
    optim_BC = torch.optim.Adam(model_BC.parameters(), lr=0.001)    
    model_CA = LSTM(input_size=lstmArgs.numVariables, hidden_layer_size=50, output_size=lstmArgs.numVariables)
    optim_CA = torch.optim.Adam(model_CA.parameters(), lr=0.001)    
    model_ALL = LSTM(input_size=lstmArgs.numVariables, hidden_layer_size=50, output_size=lstmArgs.numVariables)
    optim_ALL = torch.optim.Adam(model_ALL.parameters(), lr=0.001)      
    # generate ACN data in LSTM format
    d = LSTM_DATA(dataArgs.start, dataArgs.endTrain, dataArgs.endTest, windowSize = lstmArgs.windowSize, phaseSelection = dataArgs.phaseSelection)
    # check if data is normal to use
    print("Training data length: {0} || Testing data length: {1}.".format(len(d.dataTrain), len(d.dataTest)))
    AB = d.get_normed_sequence_byphase(numVariables = lstmArgs.numVariables, phaseType = -1)
    BC = d.get_normed_sequence_byphase(numVariables = lstmArgs.numVariables, phaseType = 0)
    CA = d.get_normed_sequence_byphase(numVariables = lstmArgs.numVariables, phaseType = 1) 
    ALL = d.get_normed_sequence_byphase(numVariables = lstmArgs.numVariables, phaseType = None) 
    
    
    
    '''
    Train learning agents
    '''
    # train or load
    if lstmArgs.isTrain:
        print("Start LSTM training between {0} - {1} || Window Size: {2} || Epochs: {3}".format(dataArgs.start, dataArgs.endTrain, lstmArgs.windowSize, lstmArgs.epochs))
        model_AB, optim_AB = train_lstm(AB["train"], model_AB, optim_AB, epochs = lstmArgs.epochs, PATH = "./runs/model/LSTM_AB.pt")
        model_BC, optim_BC = train_lstm(BC["train"], model_BC, optim_BC, epochs = lstmArgs.epochs, PATH = "./runs/model/LSTM_BC.pt")
        model_CA, optim_CA = train_lstm(CA["train"], model_CA, optim_CA, epochs = lstmArgs.epochs, PATH = "./runs/model/LSTM_CA.pt")
        model_ALL, optim_ALL = train_lstm(ALL["train"], model_ALL, optim_ALL, epochs = lstmArgs.epochs, PATH = "./runs/model/LSTM_ALL.pt")
    else:
        model_AB, optim_AB = load_model(model_AB, optim_AB, PATH = "./runs/model/LSTM_AB.pt")
        model_BC, optim_BC = load_model(model_BC, optim_BC, PATH = "./runs/model/LSTM_BC.pt")
        model_CA, optim_CA = load_model(model_CA, optim_CA, PATH = "./runs/model/LSTM_CA.pt")       
        model_ALL, optim_ALL = load_model(model_ALL, optim_ALL, PATH = "./runs/model/LSTM_ALL.pt") 
    
    def nStep_lstm_prediction(model, 
                              scaler, 
                              testInoutSequence, 
                              windowSize : int = 30,  
                              nStep : int = 1):
        """
        Based on one input sequence with specified windowSize same as model sequence dimension, predict the next n incoming 
        data.
        
        Args:
        model (nn.module): lstm model with learned parameters.
        scaler (MinMaxScaler): scaler based on the training data to convert normalized testing data to original scale.
        testInoutSequence (List[tuple(np.array, np.array)]): Inout sequence for testing.
        windowSize (int): memory size of LSTM.
        nStep (int): number of prediction made for the next n incoming data.
        
        Return:
        predictionMatrix (tuple(np.array, np.array)): prediction matrix from LSTM n-step prediction, the first np.array contains
            n-step predictions, the second np.array contains the next arriving EV's profile.
        """
        # initialize output matrix: should be a (numTestData * (nStep * EVProfile, realNextArrival)) dimension matrix.
        predictionMatrix = []
        for test_inputs, realNextArrival in testInoutSequence:
            # convert test_inputs to np.array
            test_inputs = test_inputs.numpy()
            realNextArrival = scaler.inverse_transform(realNextArrival.numpy())
            for i in range(nStep):
                seq = torch.FloatTensor(test_inputs[-windowSize:])
                with torch.no_grad():
                    model.hidden = (torch.zeros(1, 1, model.hidden_layer_size),
                                    torch.zeros(1, 1, model.hidden_layer_size))
                    #test_inputs = np.append(test_inputs, np.array(model(seq).tolist()), axis = 0)
                    test_inputs = np.append(test_inputs, model(seq).numpy(), axis = 0)
            # n step predictions are included in actural_prediction variable
            actual_predictions = scaler.inverse_transform(test_inputs[-nStep:])
            # append tuple (nStep * EVProfile, realNextArrival) to predictionMatrix
            predictionMatrix.append((actual_predictions, realNextArrival))
        return predictionMatrix
    
    
    '''
    Make predictions
    '''
    predMatrix_AB = nStep_lstm_prediction(model_AB, d.scaler, AB["test"], windowSize = lstmArgs.windowSize,  nStep = lstmArgs.nStep)
    predMatrix_BC = nStep_lstm_prediction(model_BC, d.scaler, BC["test"], windowSize = lstmArgs.windowSize,  nStep = lstmArgs.nStep)
    predMatrix_CA = nStep_lstm_prediction(model_CA, d.scaler, CA["test"], windowSize = lstmArgs.windowSize,  nStep = lstmArgs.nStep)
    predMatrix_ALL = nStep_lstm_prediction(model_ALL, d.scaler, ALL["test"], windowSize = lstmArgs.windowSize,  nStep = lstmArgs.nStep)
        
    
    
    def plot_TS(predictionMatrix, label): 
        """
        Plot time-series of EV profile and its one-step prediction from prediction matrix.
        
        Args:
        predictionMatrix (tuple(np.array, np.array)): prediction matrix from LSTM n-step prediction.
        
        Return:
            predSeq (np.array(numData * EVProfile)): one-step predicted sequence.
            realSeq (np.array(numData * EVProfile)): original testing data sequence.
        """
        
        def get_1step_prediction(predictionMatrix):
            """
            Based on the first prediction in the prediction matrix, plot the one-step predicting time-series
            against the original time-series of EV profile.
            
            Args:
            predictionMatrix (tuple(np.array, np.array)): prediction matrix from LSTM n-step prediction.
            
            Return:
            predSeq (np.array(numData * EVProfile)): one-step predicted sequence.
            realSeq (np.array(numData * EVProfile)): original testing data sequence.
            """
            realSeq = []
            predSeq = []
            for pred, label in predictionMatrix:
                realSeq.append(label[0])
                predSeq.append(pred[0])
            return np.array(realSeq), np.array(predSeq)   
        
        # get real and one-step predicted time-series ready to plot
        realSequence, predictedSequence = get_1step_prediction(predictionMatrix)
        fig, axs = plt.subplots(3)
        fig.suptitle('Time Series Analysis on {0} Phase'.format(label))
        axs[0].plot(realSequence[:, 0], label = "Real")
        axs[0].plot(predictedSequence[:, 0], label = "Pred")
        axs[0].set_title('Arrival Time')
        axs[0].set(xlabel='EV Arrivals', ylabel='Arrival Time')
        axs[0].legend(loc="upper right")
        axs[1].plot(realSequence[:, 1])
        axs[1].plot(predictedSequence[:, 1])
        axs[1].set_title('Duration')
        axs[1].set(xlabel='EV Arrivals', ylabel='Duration')
        #axs[1].legend(loc="upper right")
        axs[2].plot(realSequence[:, 2])
        axs[2].plot(predictedSequence[:, 2])
        axs[2].set_title('Energy Remaining')
        axs[2].set(xlabel='EV Arrivals', ylabel='Energy Remaining')
        #axs[2].legend(loc="upper right")
        # Hide x labels and tick labels for top plots and y ticks for right plots.
        for ax in axs.flat:
            ax.label_outer()
        plt.show()
        return realSequence, predictedSequence
    
    
    plot_TS(predMatrix_AB, "AB")
    plot_TS(predMatrix_BC, "BC")
    plot_TS(predMatrix_CA, "CA")
    plot_TS(predMatrix_ALL, "ALL")