import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import os


def plot_whole_folders(folderPath, 
                       figName = 'culmulative reward performance',
                       xlabel = 'episode', 
                       ylabel = 'culmulative reward',  
                       low = 0, 
                       up = 10):
    dataset = os.listdir(folderPath)
    plt.figure(figName)
    dataDict = {}
    for n, data in enumerate(dataset):
        if data.endswith('.npy'):
            # create name of the label
            l = data.replace('.npy', '')
            with open(os.path.join(folderPath, data), 'rb') as f:
                d = np.load(f)   
                dataDict[l] = d
            print("{0}: {1}".format(l, d))
            plt.plot(range(up - low), d[low:up], label=l)
        else:
            continue 
    plt.legend(loc = 'upper right')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(figName)    
    plt.show()
    return dataDict

########################################################
#
# Run as the main module (eg. for testing).
#
########################################################  
if __name__ == "__main__":  
    a = plot_whole_folders("runs/result", figName = 'cap=0_10_rate=10_obj_func')