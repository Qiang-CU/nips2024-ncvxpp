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


def shift(x, y, theta, eps):
    mask = torch.where(y == 1, torch.ones_like(y), torch.zeros_like(y))
    # 扩展 theta 到与 x 相同的形状
    theta_expanded = theta.unsqueeze(0).expand_as(x)
    x_new = x - eps * theta_expanded * mask.unsqueeze(1)

    return x_new

def compute_acc(pred, target):

    pred_flat = pred.view(-1)
    pred_labels = torch.where(pred_flat >= 0, torch.tensor(1), torch.tensor(-1))
    correct = (pred_labels == target).sum().item()
    accuracy = correct / target.size(0)

    return accuracy


class Simulation(object):

    def __init__(self, device, lambda_reg, a0, a1, num_epoch, grd_type, c, batch, K=1):
        self.K = K
        self.iter_ss = 0
        self.batch_size = batch
        self.num_input = 10
        self.num_output = 1
        self.num_epoch = num_epoch
        self.cur_iter = 0
        self.record = {'iter_ss': [], 'grd': [], 'loss': [], 'train_acc': [], 'test_acc': []}
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

        # for lazy deployment
        self.num_max_samples = self.num_iter * self.batch_size
        self.log_sampl_max = torch.log10(self.num_max_samples)
        self.record_time = create_sampling_time(self.log_sampl_max, True)

        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.a0 / (self.K * np.sqrt(self.num_iter)))

    def init_model(self):
        for param in self.model.parameters():
            if len(param.shape) > 1:  # 如果是权重矩阵
                nn.init.normal_(param, mean = 0, std = 1) * 10   # 使用正态分布进行初始化
            else:                      # 如果是偏置向量
                nn.init.normal_(param, mean = 0, std = 1) * 10# 使用常数初始化

    def save(self, eps):
        save_dir = './output/'
        file_name = save_dir + f'eps{eps}-{self.grd_type}-batch{self.batch_size}-K{self.K}-res{self.rank}.npy'
        np.save(file_name, self.record)

    def load_data(self):
        X_full, Y_full = load_data(file_loc='data.csv')
        self.train_X, self.test_X, self.train_Y, self.test_Y = train_test_split(X_full, Y_full, test_size=0.2, random_state=40)

        # 在training data中随机选择要反转的标签，并将它们的值反转
        # 考虑训练过程中又一些outliers, 因此只反转train data
        num_to_flip = int(len(self.train_X) * 0.1)
        flip_indices = np.random.choice(len(self.train_X), num_to_flip, replace=False)
        self.train_Y[flip_indices] = - self.train_Y[flip_indices]


        traindata = custom_dataset(self.train_X, self.train_Y)
        testdata = custom_dataset(self.test_X, self.test_Y)

        if self.grd_type == 'gd':
            # exact gradient
            self.train_loader = DataLoader(dataset = traindata, batch_size=len(self.train_X), shuffle = False)
            self.test_loader = DataLoader(dataset = testdata, batch_size=len(self.test_X), shuffle = False)
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr)
        
        elif self.grd_type == 'sgd':
            self.train_loader = DataLoader(dataset = traindata, batch_size=self.batch_size, shuffle = True)
            self.test_loader = DataLoader(dataset = testdata, batch_size=self.batch_size, shuffle = False)
            # self.optimizer = TimeVaryingSGD(self.model.parameters(), a0=self.a0, a1=self.a1)
            
        else:
            raise ValueError('Invalid gradient type')


    def test(self, theta, eps):
        self.model.eval()
        with torch.no_grad():
            X_full_shift = shift(self.test_X, self.test_Y, theta, eps)
            X_full_shift.to(self.device)
            pred = self.model(X_full_shift)
            test_y = self.test_Y.to(self.device)
            acc = compute_acc(pred, test_y)

            self.record['test_acc'].append(acc)
    
    def train(self, epoch, eps):
        self.model.train() #训练模式：启用 BatchNormalization 和 Dropout
        self.deploy_model = torch.cat([param.view(-1) for name, param in self.model.named_parameters() if 'bias' not in name])

        for batch, (X, y) in enumerate(self.train_loader):
            self.cur_iter += 1
            self.iter_ss += self.batch_size

            if self.cur_iter % self.K == 0: #lazy deployment
                self.deploy_model = torch.cat([param.view(-1) for name, param in self.model.named_parameters() if 'bias' not in name])


            X = shift(X, y, self.deploy_model, eps).to(self.device)

            # Compute prediction error
            # Backpropagation
            self.optimizer.zero_grad()

            pred = self.model(X)
            target = Variable(y.to(self.device))
            loss = self.loss_fn(pred, target)

            loss.backward()
            self.optimizer.step()

            ## record
            if self.cur_iter in self.sample_time:
                self.record['iter_ss'].append(self.iter_ss)
                self.optimizer.zero_grad()

                # theta2 = torch.cat([param.view(-1) for name, param in self.model.named_parameters() if 'bias' not in name])
                X_full_shift = shift(self.train_X, self.train_Y, self.deploy_model, eps)
                X_full_shift.to(self.device)
                pred2 = self.model(X_full_shift)
                train_y = self.train_Y.to(self.device)
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
                self.test(self.deploy_model, eps)

            if self.cur_iter % 50 == 0 and self.rank == 0:
                print(f'{self.grd_type}-GD Running: Eps: {eps}, Epoch: {epoch}, '
                        f'Sample Iter = {self.iter_ss}, '
                        f'loss: {np.round(loss.item(), 7)}, '
                        f'grd norm: {np.round(self.record["grd"][-1], 7)} '
                        f'train acc: {self.record["train_acc"][-1]}')
                sys.stdout.flush()
        
    def fit(self, eps):
        for epoch in range(self.num_epoch):
            self.train(epoch, eps)
        self.save(eps)

    


if __name__ == '__main__':

    def simu2():
        eps = 2
        K_list = [5] #5, 10, 20
        batch_size = 1
        device = "cuda" if torch.cuda.is_available() else "cpu"
        lambda_reg = 1e-3
        c = 0.1
        num_epoch = int(3*1e3)
        a0 = 500
        a1 = 0
        grd_type = 'sgd'

        # lazy deployment
        for K in K_list:
            instance = Simulation(device, lambda_reg, a0, a1, num_epoch, grd_type, c, batch=batch_size, K=K)
            instance.fit(eps)

        # greedy deployment
        # instance = Simulation(device, lambda_reg, a0, a1, num_epoch, grd_type, c, batch=25, K=1)
        # instance.fit(eps)


    simu2()


