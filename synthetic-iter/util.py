import numpy as np 
import os
import torch 

def create_sampling_time(logMaxIter, log_scale=True):
    """
        Function: create the sampling time according to geometric or linear scale
    """
    num_points = int(2000)
    maxIter = int(10**logMaxIter)
    if log_scale:
        sample_num = np.geomspace(1, 10**logMaxIter, num_points, endpoint=False, dtype=int)
    else:
        sample_num = np.arange(0, maxIter, step=(maxIter)/num_points, dtype=int) 
    return sample_num

def compute_acc(pred, target):
    pred_flat = pred.view(-1)
    pred_labels = torch.where(pred_flat >= 0, torch.tensor(1), torch.tensor(-1))
    correct = (pred_labels == target).sum().item()
    accuracy = correct / target.size(0)
    return accuracy

def get_num_epoch(log_max_iter, train_loader, batch):
    """
        Get the number of epochs for training, according to size of training data, batch and etc.
    """
    num_train_samples = len(train_loader)
    num_iter = int(10 ** log_max_iter)
    num_epoch = int(num_iter * batch / num_train_samples)  + 1
    return num_epoch, num_iter