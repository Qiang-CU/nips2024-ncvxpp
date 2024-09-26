from dataset import load_data, custom_dataset
from model import LinearLayerWithBias, BCELossWithL2, TimeVaryingSGD
from util import create_sampling_time

import sys
import torch 
import numpy as np
import torch.nn as nn 
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from mpi4py import MPI


def shift(x, theta, eps):
    x = x - eps * theta
    return x 

def compute_acc(pred, target):
    pred = (pred > 0.5).float().squeeze() # pred dim is (n, 1) -> (n)
    correct = (pred == target).sum().item()
    ratio = correct / len(target)
    return ratio


class Simulation(object):

    def __init__(self, device, lambda_reg, a0, a1, num_epoch, grd_type, c):
        self.batch_size = 1
        self.num_input = 20
        self.num_output = 1
        self.num_epoch = num_epoch
        self.cur_iter = 0
        self.record = {'iter': [], 'grd': [], 'loss': [], 'train_acc': [], 'test_acc': []}
        self.a0, self.a1 = a0, a1        
        self.grd_type = grd_type
        
        self.model = LinearLayerWithBias(self.num_input, self.num_output, c).to(device)
        self.init_model()
        self.loss_fn = BCELossWithL2(self.model, lambda_reg=lambda_reg)
        self.load_data() #optimizer is set while loading data

        # MPI setting
        self.device = device
        self.comm = MPI.COMM_WORLD
        self.size = self.comm.Get_size()
        self.rank = self.comm.Get_rank()

        self.size = len(self.train_loader)
        self.num_iter = torch.tensor(self.num_epoch * self.size)
        self.log_maxiter = torch.log10(self.num_iter)
        self.sample_time = create_sampling_time(self.log_maxiter, True)

    def init_model(self):
        for param in self.model.parameters():
            if len(param.shape) > 1:  # 如果是权重矩阵
                nn.init.normal_(param, mean = 1, std = 1)    # 使用正态分布进行初始化
            else:                      # 如果是偏置向量
                nn.init.normal_(param, mean = 1, std = 1) # 使用常数初始化

    def save(self, eps):
        save_dir = './output/'
        file_name = save_dir + f'eps{eps}-{self.grd_type}-res{self.rank}.npy'
        np.save(file_name, self.record)

    def load_data(self):
        X_full, Y_full = load_data(file_loc='data.csv')
        self.train_X, self.test_X, self.train_Y, self.test_Y = train_test_split(X_full, Y_full, test_size=0.3, random_state=42)
        traindata = custom_dataset(self.train_X, self.train_Y)
        testdata = custom_dataset(self.test_X, self.test_Y)

        if self.grd_type == 'gd':
            # exact gradient
            self.train_loader = DataLoader(dataset = traindata, batch_size=len(self.train_X), shuffle = False)
            self.test_loader = DataLoader(dataset = testdata, batch_size=len(self.test_X), shuffle = False)
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=1e-1)
        
        elif self.grd_type == 'sgd':
            self.train_loader = DataLoader(dataset = traindata, batch_size=self.batch_size, shuffle = True)
            self.test_loader = DataLoader(dataset = testdata, batch_size=self.batch_size, shuffle = False)
            self.optimizer = TimeVaryingSGD(self.model.parameters(), a0=self.a0, a1=self.a1)
            
        else:
            raise ValueError('Invalid gradient type')


    def test(self, theta, eps):
        self.model.eval()
        with torch.no_grad():
            X_full_shift = shift(self.test_X, theta, eps)
            X_full_shift.to(self.device)
            pred = self.model(X_full_shift)
            test_y = self.test_Y.to(device)
            acc = compute_acc(pred, test_y)

            self.record['test_acc'].append(acc)
    
    def train(self, epoch, eps):
        # self.model.train() #训练模式：启用 BatchNormalization 和 Dropout

        for batch, (X, y) in enumerate(self.train_loader):
            theta = torch.cat([param.view(-1) for name, param in self.model.named_parameters() if 'bias' not in name])
            X = shift(X, theta, eps).to(self.device)

            # Compute prediction error
            pred = self.model(X)
            target = Variable(y.to(self.device))
            loss = self.loss_fn(pred, target)

            # Backpropagation
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # record
            self.cur_iter += 1
            if self.cur_iter in self.sample_time:
                self.record['iter'].append(self.cur_iter)
                self.optimizer.zero_grad()

                theta2 = torch.cat([param.view(-1) for name, param in self.model.named_parameters() if 'bias' not in name])
                X_full_shift = shift(self.train_X, theta2, eps)
                X_full_shift.to(self.device)
                pred2 = self.model(X_full_shift)
                train_y = self.train_Y.to(device)
                acc = compute_acc(pred2, train_y)

                total_loss = self.loss_fn(pred2, train_y)
                total_loss.backward()

                # note: 注意这里的计算方式
                full_grad = torch.cat([param.grad.view(-1) for name, param in self.model.named_parameters()])
                grd_norm = full_grad.norm(2) ** 2
                
                self.record['grd'].append(grd_norm.item())
                self.record['loss'].append(total_loss.item())
                self.record['train_acc'].append(acc)

                # test
                self.test(theta, eps)

            if self.cur_iter % 50 == 0 and self.rank == 0:
                print(f'{self.grd_type}-GD Running: Eps: {eps}, Epoch: {epoch}, '
                        f'Iter = {self.cur_iter}, '
                        f'loss: {np.round(loss.item(), 4)}, '
                        f'grd norm: {np.round(self.record["grd"][-1], 4)}')
                sys.stdout.flush()
        
    def fit(self, eps):
        for epoch in range(self.num_epoch):
            self.train(epoch, eps)
        self.save(eps)

    


if __name__ == '__main__':
    
    eps_list = [0, 0.1, 0.5, 2, 10] #,
    device = "cuda" if torch.cuda.is_available() else "cpu"
    lambda_reg = 1e-3
    a0 = 20
    a1 = 5000
    
    
    num_epoch = int(1e5)
    grd_type_list = ['gd']
    c = 2

    for grd_type in grd_type_list:
        for eps in eps_list:
            instance = Simulation(device, lambda_reg, a0, a1, num_epoch, grd_type, c)
            instance.fit(eps)
