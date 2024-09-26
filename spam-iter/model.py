import torch 
from torch import nn 
import torch.nn.functional as F


class SpamNN(nn.Module):
    def __init__(self):
        super(SpamNN, self).__init__()
        self.fc1 = nn.Linear(in_features=57, out_features=50)

        self.fc2 = nn.Linear(in_features=50, out_features=10)
        self.fc3 = nn.Linear(in_features=10, out_features=1)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh() 

        self.init_model()
    
    def forward(self, x):
        x = self.tanh(self.fc1(x))
        x = self.tanh(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        return x 
    
    def init_model(self):
        for param in self.parameters():
            if len(param.shape) > 1:  # 如果是权重矩阵
                # nn.init.constant_(param, 1)    # 使用正态分布进行初始化
                nn.init.normal_(param, mean=0, std=1)
            else:                      # 如果是偏置向量
                nn.init.constant_(param, 0) # 使用常数初始化


class BCELossWithL2(nn.Module):    
    def __init__(self, model, lambda_reg=1e-3):
        super(BCELossWithL2, self).__init__()
        self.model = model
        self.lambda_reg = lambda_reg  # L2 正则化的系数
        self.bce_loss = nn.BCEWithLogitsLoss() # with sigmoid 

    def forward(self, outputs, labels):
        labels = torch.unsqueeze(labels, 1) 
        loss = self.bce_loss(outputs, labels)  # 损失函数的计算
        l2_reg = 0  # 初始化 L2 正则化项
        for param in self.model.parameters():  # 遍历模型的参数
            l2_reg += torch.norm(param, p=2) ** 2  # 计算参数的 L2 范数的平方并累加
        loss += 0.5 * self.lambda_reg * l2_reg  # 添加 L2 正则化项
        return loss