import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np


class LinearLayerWithBias(nn.Module):
    def __init__(self, input_dim, output_dim, c=1):
        super(LinearLayerWithBias, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.c = c

    def forward(self, x):
        return self.linear(x) * self.c


class BCELossWithL2(nn.Module):    
    def __init__(self, model, lambda_reg=1e-3):
        super(BCELossWithL2, self).__init__()
        self.model = model
        self.lambda_reg = lambda_reg  # L2 正则化的系数

    def forward(self, outputs, labels):
        labels = torch.unsqueeze(labels, 1) 
        loss = (1 / (1 + torch.exp(outputs * labels))).mean()
        
        l2_reg = 0  # 初始化 L2 正则化项
        for param in self.model.parameters():  # 遍历模型的参数
            l2_reg += torch.norm(param, p=2) ** 2  # 计算参数的 L2 范数的平方并累加
        loss += 0.5 * self.lambda_reg * l2_reg  # 添加 L2 正则化项
        return loss


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