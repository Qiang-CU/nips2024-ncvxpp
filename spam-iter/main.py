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


class Simulation:
    def __init__(self, eps, batch, file_loc, num_epoch, a0, a1, log_maxiter, device):
        self.eps = eps
        self.batch = batch
        self.file_loc = file_loc
        self.num_epoch = num_epoch
        self.a0, self.a1 = a0, a1
        self.device = device
        self.sample_time = create_sampling_time(log_maxiter, True)
        self.record = {'iter': [], 'grd': [], 'train_loss': [], 'train_acc': [], 'test_acc': [], 'test_loss': []}
        self.iter = 0

        # model & optimizer
        self.model = SpamNN().to(device)
        self.loss_fn = BCELossWithL2(self.model, lambda_reg=1e-4)
        self.prepare_data()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.a0 / (self.a1 + np.sqrt(self.num_epoch * len(self.train_loader))))


        params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"The model has {params} trainable parameters.")
        exit(0)

        # MPI Setting
        self.comm = MPI.COMM_WORLD
        self.size = self.comm.Get_size()
        self.rank = self.comm.Get_rank()

    def prepare_data(self):
        X_full, Y_full = load_spamdata(self.file_loc)
        self.train_X, self.test_X, self.train_Y, self.test_Y = train_test_split(X_full, Y_full, test_size=0.2, random_state=42)
        self.train_loader = DataLoader(custom_dataset(self.train_X, self.train_Y), batch_size=self.batch, shuffle=True)
        self.test_loader = DataLoader(custom_dataset(self.test_X, self.test_Y), batch_size=self.batch, shuffle=False)

        # set max_iter
        self.num_iter = self.num_epoch * len(self.train_loader)
        self.log_maxiter = np.log10(self.num_iter)

    def save(self):
        save_dir = './output/'
        np.save(save_dir + f'eps{self.eps}-res{self.rank}.npy', self.record)

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
        self.deploy_model = None
        for idx, (X, Y) in enumerate(self.train_loader):
            self.iter += 1

            #TODO: squeeze all data
            # shift data
            self.optimizer.zero_grad()
            XX = X.clone().detach().requires_grad_(True)
            output = self.model(XX)
            output.backward(torch.ones_like(output))
            X_shift = self.shift(X, Y, XX.grad, self.eps).to(self.device)
            
            #update
            self.optimizer.zero_grad()
            target = Y.to(self.device)
            pred = self.model(X_shift)
            loss = self.loss_fn(pred, target)
            loss.backward()
            self.optimizer.step()

            if self.iter in self.sample_time:
                self.record['iter'].append(self.iter)

                # get shift feature
                self.optimizer.zero_grad()
                XX = self.train_X.clone().detach().requires_grad_(True)
                output = self.model(XX)
                output.backward(torch.ones_like(output)) 
                X_full_shift = self.shift(self.train_X, self.train_Y, XX.grad, eps).to(self.device) 

                # get loss and acc based on shifted data
                self.optimizer.zero_grad()
                pred_train = self.model(X_full_shift)
                total_loss = self.loss_fn(pred_train, self.train_Y.to(self.device))
                acc = self.compute_acc(pred_train, self.train_Y)
                total_loss.backward()

                # get gradient norm based on shifted data
                grd = torch.cat([param.grad.view(-1) if len(param.shape) > 1 else param.grad for name, param in self.model.named_parameters()])
                grd_norm = grd.norm(2) ** 2

                self.record['grd'].append(grd_norm.item())
                self.record['train_loss'].append(total_loss.item())
                self.record['train_acc'].append(acc)

                #
                self.test(eps)

            if self.iter % 50 == 0 and self.rank == 0:
                print(f"EPS: {eps} Epoch: {epoch} Iter: {self.iter} loss: {loss.item():>5f} grd norm: {self.record['grd'][-1]:>5f}")
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

    # setup_seed(MPI.COMM_WORLD.Get_rank() + 1)
    start_time = time.time()
    batch = 4
    file_loc = './dataset/spambase.data'
    num_epoch = int(1e2)
    a0, a1 = 200, 0
    log_maxiter = 5
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    eps_list = [10000, ] # [0, 10, 100]

    for eps in eps_list:
        sim = Simulation(eps, batch, file_loc, num_epoch, a0, a1, log_maxiter, device)
        sim.fit()

    end_time = time.time()
    print(f'batch{batch}, time cost: {end_time - start_time}s')
