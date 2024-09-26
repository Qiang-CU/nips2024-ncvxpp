import sys
import torch
import numpy as np
import random
from mpi4py import MPI
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset


from dataset import load_spamdata, custom_dataset
from model import SpamNN, BCELossWithL2
from sgd import TimeVaryingSGD
from utils import create_sampling_time
import time 

def cal_grd_norm(model):
    total_norm = 0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm
    return total_norm

class Simulation:
    def __init__(self, eps, file_loc, a0, a1, log_maxiter, device, batch=1, K=1):
        self.K = K
        self.eps = eps
        self.batch = batch
        self.file_loc = file_loc
        self.a0, self.a1 = a0, a1
        self.device = device
        # load data
        self.prepare_data()

        self.max_sample_iter = int(10 ** log_maxiter)
        self.max_iter = int(self.max_sample_iter / self.batch )
        self.num_epoch = int(self.max_iter / len(self.train_X) * self.batch + 1) #多加一个epoch，防止不够
        self.sample_time = create_sampling_time(log_maxiter, True)


        self.record = {'iter_ss': [], 'grd': [], 'train_loss': [], 'train_acc': [], 'test_acc': [], 'test_loss': []}
        self.iter = 0
        self.iter_ss = 0

        # model & optimizer
        self.model = SpamNN().to(device)
        self.loss_fn = BCELossWithL2(self.model, lambda_reg=1e-4)
        # self.optimizer = TimeVaryingSGD(self.model.parameters(), a0=a0, a1=a1)
        
        
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.a0 / (self.a1 + self.K * np.sqrt(self.num_epoch * len(self.train_loader))))

        # MPI Setting
        self.comm = MPI.COMM_WORLD
        self.size = self.comm.Get_size()
        self.rank = self.comm.Get_rank()

    def prepare_data(self):
        X_full, Y_full = load_spamdata(self.file_loc)
        self.train_X, self.test_X, self.train_Y, self.test_Y = train_test_split(X_full, Y_full, test_size=0.2, random_state=42)
        self.train_loader = DataLoader(custom_dataset(self.train_X, self.train_Y), batch_size=self.batch, shuffle=True)
        self.test_loader = DataLoader(custom_dataset(self.test_X, self.test_Y), batch_size=self.batch, shuffle=False)

    def save(self):
        save_dir = './output/'
        np.save(save_dir + f'eps{self.eps}-batch{self.batch}-K{self.K}-res{self.rank}.npy', self.record)
        torch.save(self.model, save_dir + f'eps{self.eps}-batch{self.batch}-K{self.K}-model{self.rank}.pt')


    def shift(self, x, y, approx, eps): #TODO
        mask = torch.where(y == 1, torch.ones_like(y), torch.zeros_like(y))
        x_new = x - eps * approx * mask.unsqueeze(1) #TODO: 可以对特定的feature进行shift
        return x_new

    def fit(self):
        for epoch in range(self.num_epoch):
            self.train(epoch)
        self.save()

    def test(self, eps):

        XX = self.test_X.clone().detach().requires_grad_(True)
        output = self.model(XX)
        output.backward(torch.ones_like(output))
        X_test_shift = self.shift(self.test_X, self.test_Y, XX.grad, eps).to(self.device)

        # self.model.eval()
        pred = self.model(X_test_shift)
        loss = self.loss_fn(pred, self.test_Y.to(self.device))
        acc = self.compute_acc(pred, self.test_Y)
        self.record['test_acc'].append(acc)
        self.record['test_loss'].append(loss.item())
    
    def compute_acc(self, pred, target):
        pred = (pred > 0.5).float().squeeze() # pred dim is (n, 1) -> (n)
        correct = (pred == target).sum().item()
        ratio = correct / len(target)
        return ratio

    def train(self, epoch):
        self.model.train()
        self.deploy_model = torch.zeros_like(self.train_X)

        for batch_id, (X, Y, indices) in enumerate(self.train_loader):

            # deploy model
            if self.iter % self.K == 0: #lazy deployment
                self.optimizer.zero_grad()
                XX = self.train_X.clone().detach().requires_grad_(True)
                output = self.model(XX)
                output.backward(torch.ones_like(output)) 
                self.deploy_model = XX.grad.clone().detach()
            
            self.iter += 1
            self.iter_ss += self.batch

            #TODO: squeeze all data
            # shift data
            self.optimizer.zero_grad()
            X_shift = self.shift(X, Y, self.deploy_model[indices], self.eps).to(self.device)
            
            #update
            self.optimizer.zero_grad()
            target = Y.to(self.device)
            pred = self.model(X_shift)
            loss = self.loss_fn(pred, target)
            loss.backward()
            self.optimizer.step()

            ## record 
            if self.iter_ss in self.sample_time:
                self.record['iter_ss'].append(self.iter_ss)

                # get shift feature
                X_full_shift = self.shift(self.train_X, self.train_Y, self.deploy_model, eps).to(self.device) 

                # get loss and acc based on shifted data
                self.optimizer.zero_grad()
                pred_train = self.model(X_full_shift)
                total_loss = self.loss_fn(pred_train, self.train_Y.to(self.device))
                acc = self.compute_acc(pred_train, self.train_Y)
                total_loss.backward()

                #get gradient norm based on shifted data
                grd_norm = cal_grd_norm(self.model)


                self.record['grd'].append(grd_norm)
                self.record['train_loss'].append(total_loss.item())
                self.record['train_acc'].append(acc)

                #
                self.test(eps)

            if self.iter % 50 == 0 and self.rank == 0:
                print(f"EPS: {eps}, K:{self.K}, batch:{self.batch}, Epoch: {epoch}/{self.num_epoch} No of Sample: {self.iter_ss} loss: {loss.item():>5f} grd norm: {self.record['grd'][-1]:>5f}")
                sys.stdout.flush()
        


if __name__ == '__main__':
    def setup_seed(seed):
        """ Setup random seed to avoid the impact of randomness.

        Parameters
        ----------
        seed : int
            random seed.
        """
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True

    setup_seed(MPI.COMM_WORLD.Get_rank() + 1)

    eps_list = [10, 10**4] #
    file_loc = './dataset/spambase.data'
    log_maxiter = 6
    a0, a1 = 300, 0
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    for eps in eps_list:
        # Lazy Deployment
        sim = Simulation(eps, file_loc, a0, a1, log_maxiter, device, batch=4, K=4)
        sim.fit()

        sim = Simulation(eps, file_loc, a0, a1, log_maxiter, device, batch=1, K=16)
        sim.fit()

        # Greedy Deployment
        sim_greedy = Simulation(eps, file_loc, a0, a1, log_maxiter, device, batch=16, K=1)
        sim_greedy.fit()


