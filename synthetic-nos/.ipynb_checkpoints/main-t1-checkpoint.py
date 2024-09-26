import sys
import torch 
import numpy as np
import torch.nn as nn 
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from mpi4py import MPI
from dataset import load_data, custom_dataset
from model import LinearLayerWithBias, BCELossWithL2, TimeVaryingSGD
from util import create_sampling_time
from mpi4py import MPI

def shift(x, theta, eps):
    x = x - eps * theta
    return x 

def init_model(model):
    for param in model.parameters():
        if len(param.shape) > 1:  # 如果是权重矩阵
            nn.init.normal_(param)    # 使用正态分布进行初始化
        else:                      # 如果是偏置向量
            nn.init.normal_(param) # 使用常数初始化

def save(record, eps, rank):
    save_dir = './output/'
    np.save(save_dir + f'eps{eps}-res{rank}.npy', record)

def train(model, train_loader, test_X, train_Y, device, loss_fn, optimizer, sample_time, eps, num_epoch, size, rank):
    model.train()
    cur_iter = 0
    record = {'iter': [], 'grd': [], 'loss': []}

    for epoch in range(num_epoch):
        for batch, (X, y) in enumerate(train_loader):
            theta = torch.cat([param for name, param in model.named_parameters() if 'bias' not in name])
            X = shift(X, theta, eps).to(device)

            # Compute prediction error
            pred = model(X)
            target = Variable(y.to(device))
            loss = loss_fn(pred, target)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # record
            cur_iter += 1
            if cur_iter in sample_time:
                record['iter'].append(cur_iter)

                optimizer.zero_grad()
                X_full_shift = shift(train_X, theta, eps)
                X_full_shift.to(device)
                train_y = train_Y.to(device)
                
                totoal_loss = loss_fn(model(X_full_shift), train_y)
                totoal_loss.backward()
                full_grad = torch.cat([param.squeeze(0) if len(param.shape) > 1 else param for name, param in model.named_parameters()])

                total_norm = full_grad.norm(2)**2
                record['grd'].append(total_norm.item())
                record['loss'].append(totoal_loss.item())

            if cur_iter % 50 == 0 and rank == 0:
                print(f"EPS: {eps} Epoch: {epoch} Iter: {cur_iter} loss: {loss.item():>5f} grd norm: {record['grd'][-1]:>5f}")
                sys.stdout.flush()
    save(record, eps, rank)

def fit(train_X, train_Y, test_X, test_Y, batch_size, lambda_reg, a0, a1, device, eps_list, num_epoch, rank):
    traindata = custom_dataset(train_X, train_Y)
    testdata = custom_dataset(test_X, test_Y)

    train_loader = DataLoader(dataset=traindata, batch_size=batch_size, shuffle=True)

    size = len(train_loader)
    num_iter = torch.tensor(num_epoch * size)
    log_maxiter = torch.log10(num_iter)
    sample_time = create_sampling_time(log_maxiter, True)

    for eps in eps_list:
        model = LinearLayerWithBias(2, 1).to(device)
        init_model(model)
        loss_fn = BCELossWithL2(model, lambda_reg=lambda_reg)
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)

        for epoch in range(num_epoch):
            train(model, train_loader, test_X, train_Y, device, loss_fn, optimizer, sample_time, eps, num_epoch, size, rank)

if __name__ == '__main__':
    eps_list = [0]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    lambda_reg = 0
    a0 = 10
    a1 = 100
    batch_size = 1
    num_epoch = 30

    X_full, Y_full = load_data(file_loc='data.csv')
    train_X, test_X, train_Y, test_Y = train_test_split(X_full, Y_full, test_size=0.2,)

    fit(train_X, train_Y, test_X, test_Y, batch_size, lambda_reg, a0, a1, device, eps_list, num_epoch, MPI.COMM_WORLD.Get_rank())
