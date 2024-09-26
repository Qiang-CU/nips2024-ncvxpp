import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np


class TimeVaryingSGD(optim.SGD):
    def __init__(self, params, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False, a0=1, a1=1):
        super(TimeVaryingSGD, self).__init__(params, momentum, dampening, weight_decay, nesterov)
        self.a0 = a0
        self.a1 = a1
        self.iteration = 0
        
    def step(self, closure=None):
        self.iteration += 1
        # Update learning rate
        self.lr = self.a0 / (self.a1 + np.sqrt(self.iteration - 1))
        for group in self.param_groups:
            group['lr'] = self.lr
        
        # Call the original SGD's step function
        super(TimeVaryingSGD, self).step(closure)