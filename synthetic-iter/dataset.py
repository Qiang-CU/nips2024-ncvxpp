import torch as t
import pandas as pd
import torch.nn as nn 
import numpy as np
from torch.utils.data import Dataset
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

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


def load_raw_data(file_loc = 'data.csv'):
    """
        function: load raw data from csv file and preprocess it (balance classes and shuffle arrays)
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


def load_data(file_loc, batch, flip_para={'flip_flag': False, 'flip_ratio': 0.1}, ):
    """
        Function: load data from specified loaction
        filp_para: dict{'flip_flag': True/False, 'flip_ratio': 0.1}
        batch: 'full' or int
    """
    X_full, Y_full = load_raw_data(file_loc='data.csv')
    train_X, test_X, train_Y, test_Y = train_test_split(X_full, Y_full, test_size=0.2, random_state=40)
    
    # 在training data中随机选择要反转的标签，并将它们的值反转
    # 考虑训练过程中又一些outliers, 因此只反转train data
    num_to_flip = int(len(train_X) * flip_para['flip_ratio'])
    flip_indices = np.random.choice(len(train_X), num_to_flip, replace=False)
    train_Y[flip_indices] = - train_Y[flip_indices]
    traindata = custom_dataset(train_X, train_Y)
    testdata = custom_dataset(test_X, test_Y)

    if batch == 'full':
        train_loader = DataLoader(dataset = traindata, batch_size=len(train_X), shuffle = False)
        test_loader = DataLoader(dataset = testdata, batch_size=len(test_X), shuffle = False)
    elif isinstance(batch, int):
        train_loader = DataLoader(dataset = traindata, batch_size=batch, shuffle = True)
        test_loader = DataLoader(dataset = testdata, batch_size=batch, shuffle = False)
    else:
        raise ValueError('Invalid batch size')
    return train_loader, test_loader, train_X, test_X, train_Y, test_Y, 