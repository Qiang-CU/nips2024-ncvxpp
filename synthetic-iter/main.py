from dataset import load_data, custom_dataset
from model import LinearLayerWithBias, BCELossWithL2, TimeVaryingSGD
from util import create_sampling_time, get_num_epoch, compute_acc

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

class BasicSimu(object):
    def __init__(self, device, lambda_reg, c, batch, a0, a1):
        self.device = device
        self.batch_size = batch
        self.num_input = 2
        self.num_output = 1
        self.cur_iter = 0
        self.a0, self.a1 = a0, a1
        self.record = {'iter': [], 'grd': [], 'loss': [], 'train_acc': [], 'test_acc': [], 'train_acc_unshift':[], 'test_acc_unshift':[]}

        # MPI setting
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
    

    def SetOpt_CstStep_SGD(self, batch, regularizer, c, num_iter):
        self.model = LinearLayerWithBias(self.num_input, self.num_output, c).to(self.device)       
        self.train_loader, self.test_loader, self.train_X, self.test_X, self.train_Y, self.test_Y \
                        =  load_data(file_loc = 'circle_data.csv',  batch=batch, flip_para = {'flip_flag': True, 'flip_ratio': 0.1}) 
        self.loss_fn = BCELossWithL2(self.model, lambda_reg=regularizer)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr = self.a0/(np.sqrt(num_iter)))

    # def SetOpt_VaryingStep_SGD(self, batch, regularizer, c):
    #     self.model = LinearLayerWithBias(self.num_input, self.num_output, c).to(self.device)       
    #     self.train_loader, self.test_loader, self.train_X, self.test_X, self.train_Y, self.test_Y \
    #                     =  load_data(file_loc = 'data.csv', flip_para = {'flip_flag': True, 'flip_ratio': 0.1}, batch=batch) 
    #     self.loss_fn = BCELossWithL2(self.model, lambda_reg=regularizer)
    #     self.optimizer = torch.optim.TimeVaryingSGD(self.model.parameters(), a0, a1)
    #     self.num_epoch, self.num_iter = get_num_epoch(self.log_maxiter, self.train_loader, self.batch_size)
    
    def save(self, eps):
        save_dir = './output/'
        file_name = save_dir + f'eps{eps}-batch{self.batch}-res{self.rank}.npy'
        np.save(file_name, self.record)





class Simulation(BasicSimu):

    def __init__(self, device, regularizer, a0, a1, grd_type, c, lr, batch, log_maxiter):
        super().__init__(device, regularizer, c, batch, a0, a1)
        self.device = device
        self.batch = batch
        self.num_input = 10
        self.num_output = 1
        self.cur_iter = 0
        self.record = {'iter': [], 'grd': [], 'loss': [], 'train_acc': [], 'test_acc': [], 'train_acc_unshift':[], 'test_acc_unshift':[]}
        self.log_maxiter = log_maxiter
        self.grd_type = grd_type

        # MPI setting
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()

        # running settings
        self.sample_time = create_sampling_time(self.log_maxiter, True)
        self.num_iter = int(10 ** self.log_maxiter)
        self.SetOpt_CstStep_SGD(batch, regularizer, c, self.num_iter)
        self.num_epoch, self.num_iter = get_num_epoch(self.log_maxiter, self.train_loader, batch)

    def test(self, deploy_model, eps):
        self.model.eval()
        with torch.no_grad():
            X_full_shift = shift(self.test_X, self.test_Y, deploy_model, eps)
            X_full_shift.to(self.device)
            pred = self.model(X_full_shift)
            test_y = self.test_Y.to(self.device)
            acc = compute_acc(pred, test_y)

            unshift_pred = self.model(self.test_X.to(self.device))
            unshift_acc = compute_acc(unshift_pred, test_y)
            self.record['test_acc'].append(acc)
            self.record['test_acc_unshift'].append(unshift_acc)
    
    def train(self, epoch, eps):
        self.model.train() #训练模式：启用 BatchNormalization 和 Dropout

        for batch, (X, y) in enumerate(self.train_loader):
            
            self.deploy_model = torch.cat([param.view(-1) for name, param in self.model.named_parameters() if 'bias' not in name])
            X = shift(X, y, self.deploy_model, eps).to(self.device)
            self.optimizer.zero_grad()

            pred = self.model(X)
            target = Variable(y.to(self.device))
            loss = self.loss_fn(pred, target)
            loss.backward()
            self.optimizer.step()

            self.cur_iter += 1
            if self.cur_iter in self.sample_time:
                self.record['iter'].append(self.cur_iter)
                self.optimizer.zero_grad()

                X_full_shift = shift(self.train_X, self.train_Y, self.deploy_model, eps)
                X_full_shift.to(self.device)
                pred2 = self.model(X_full_shift)
                train_y = self.train_Y.to(self.device)
                acc = compute_acc(pred2, train_y)

                total_loss = self.loss_fn(pred2, train_y)
                total_loss.backward()

                ## unshift data set acc
                pred_unshift = self.model(self.train_X.to(self.device))
                acc_unshift = compute_acc(pred_unshift, self.train_Y)

                # note: 注意这里的计算方式
                full_grad = torch.cat([param.grad.view(-1) for name, param in self.model.named_parameters()])
                grd_norm = full_grad.norm(2) ** 2
                
                self.record['grd'].append(grd_norm.item())
                self.record['loss'].append(total_loss.item())
                self.record['train_acc'].append(acc)
                self.record['train_acc_unshift'].append(acc_unshift)

                # test
                self.test(self.deploy_model, eps)

            if self.cur_iter % 50 == 0 and self.rank == 0:
                print(f'{self.grd_type}-GD Running: Eps: {eps}, Epoch: {epoch}, '
                        f'Iter = {self.cur_iter}, '
                        f'loss: {np.round(loss.item(), 5)}, '
                        f'grd norm: {np.round(self.record["grd"][-1], 7)}, '
                        f'train acc: {np.round(self.record["train_acc"][-1], 4)}.')
                sys.stdout.flush()
        
    def fit(self, eps):
        for epoch in range(self.num_epoch):
            self.train(epoch, eps)
        self.save(eps)

        with open('log.txt', 'a') as file:
            file.write(f'eps: {eps}, rank: {self.rank}, final model: {torch.cat([param.grad.view(-1) for name, param in self.model.named_parameters()])} \n')
    


if __name__ == '__main__':
    
    def simu1():
        eps_list = [0, 0.1, 10, 100] #dimin and cnt
        device = "cuda" if torch.cuda.is_available() else "cpu"
        lambda_reg = 1e-3
        c = 0.1

        a0 = 50
        a1 = 10
        lr = 1e-2
        grd_type = 'sgd'

        for eps in eps_list:
            instance = Simulation(device, lambda_reg, a0, a1, grd_type, c, lr, batch=5, log_maxiter=5)
            instance.fit(eps)

    simu1()


