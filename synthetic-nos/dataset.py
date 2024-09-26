import torch as t
import pandas as pd
import torch.nn as nn 
import numpy as np
from torch.utils.data import Dataset
from sklearn import preprocessing


class custom_dataset(Dataset):
    """
        class: custom dataset according to the features and labels
    """
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y
    
    def __len__(self):
        return len(self.X)        

    def __getitem__(self, idx):
        return [self.X[idx], self.Y[idx]] 


def load_data(file_loc = 'data.csv'):
    """
        function: load data from csv file and preprocess it
    """
    raw_data = pd.read_csv(file_loc)
    raw_data.dropna(inplace = True)

    X_all = raw_data.drop('y', axis=1)
    X_all = preprocessing.scale(X_all)

    Y_all = np.array(raw_data['y'])
    
    # balance classes
    default_indices = np.where(Y_all == 1)[0]
    other_indices = np.where(Y_all == 0)[0]
    indices = np.concatenate((default_indices, other_indices))

    X_balanced = X_all[indices]
    Y_balanced = Y_all[indices]

    # shuffle arrays
    p = np.random.permutation(len(indices))
    X_full = t.tensor(X_balanced[p], dtype = t.float)
    Y_full = t.tensor(Y_balanced[p], dtype = t.float)

    print(f"num of features: {X_full.shape[1]}, label set are {set(Y_full.numpy())}")

    return X_full, Y_full