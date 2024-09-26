import torch
import pandas as pd
import torch.nn as nn 
import numpy as np
from torch.utils.data import Dataset
from sklearn import preprocessing
from sklearn.utils import shuffle

def load_spamdata(file_path):
    """
        功能：读取数据，处理缺失值，提取特征和标签，打乱数据，转换为torch张量
    """
    # 读取数据
    data = pd.read_csv(file_path, header=None)
    
    # 处理缺失值，丢弃含有NA的记录
    data.dropna(inplace=True)
    
    # 提取特征和标签
    X_all = data.iloc[:, :-1].values.astype(float)
    Y_all = data.iloc[:, -1].values.astype(float)
    
    # 数据打乱
    X_all, Y_all = shuffle(X_all, Y_all, random_state=42)
    
    # 转换为torch张量
    X_all_tensor = torch.tensor(X_all, dtype=torch.float32)
    Y_all_tensor = torch.tensor(Y_all, dtype=torch.float32)

    # 打印数据维度
    print("Info of Spam Dataset:")
    print("num of record:", X_all.shape[0])
    print("num of features", X_all.shape[1])    
    
    return X_all_tensor, Y_all_tensor


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
        return [self.X[idx], self.Y[idx], idx]


