import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import os



def save_model(model, optimizer, PATH = "./runs/model/LSTM_AB.pt"):
    """
    Save model parameters and optimizers states to designated folder path.
    
    Args:
    model (nn.module): lstm model with learned parameters.
    optimizer (torch.optim): optimizer with updated states.
    PATH (str): file path and name to save the learned model.
    
    Return:
    None
    """
    torch.save({'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict()}, PATH)



def load_model(model, optimizer, PATH = "./runs/model/LSTM_AB.pt", evaluation = True):
    """
    Load model parameters and optimizers states from designated folder path.
    Use model as evaluation if True.
    
    Args:
    model (nn.module): lstm model with learned parameters.
    optimizer (torch.optim): optimizer with updated states.
    PATH (str): file path and name to save the learned model.
    evaluation (bool): if True, model is ready for evaluation; if False, model is resumed for training.
    
    Return:
    model (nn.module): lstm model that has been equipped with loaded parameters.
    optimizer (torch.optim): optimizer that has been equipped with loaded states.
    """
    checkpoint = torch.load(PATH)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if evaluation:
        model.eval()
    else: # - or -
        model.train()    
    return model, optimizer


def plot_whole_folders(folderPath, 
                       figName = 'culmulative reward performance',
                       xlabel = 'episode', 
                       ylabel = 'culmulative reward',  
                       low = 0, 
                       up = 300):
    dataset = os.listdir(folderPath)
    plt.figure(figName)
    for n, data in enumerate(dataset):
        if data.endswith('.npy'):
            # create name of the label
            l = data.replace('.npy', '')
            with open(os.path.join(folderPath, data), 'rb') as f:
                d = np.load(f)   
            plt.plot(range(up - low), d[low:up], label=l)
        else:
            continue 
    plt.legend()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(figName)    
    plt.show()