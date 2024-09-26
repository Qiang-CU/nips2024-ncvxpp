import os
import numpy as np
import matplotlib.pyplot as plt

class plot_figure(object):
    def __init__(self, algo_name, xdata, ydata, dir, sub_sample=20, ):
        self.xdata = xdata
        self.ydata = ydata
        self.dir = dir
        self.algo_name = algo_name
        self.sub_sample = sub_sample
        self.num_trails, self.res, self.xvals = self.load_data()
        self.z = 1.96/np.sqrt(self.num_trails) # 95% confidence， 1.645-90%

    def load_data(self):
        file = [f for f in os.listdir(self.dir) if self.algo_name in f]
        files = [np.load(os.path.join(self.dir, f), allow_pickle=True) for f in file]
        res = np.array([f.item().get(self.ydata) for f in files])
        print(file)
        xvals = files[0].item().get(self.xdata)
        message = f""" There are {len(file)} data files, each file has {[len(r) for r in res]} data points"""
        return len(file), res, xvals

    def plot_lines(self, ax, color, line='-', label='', plot_star = False, shadow_flag=True, legend=True, ax_insert = None, range_=None):
        mean = np.mean(self.res, axis = 0)
        std = np.std(self.res, axis = 0)
        lb = np.squeeze(mean - self.z * std / np.sqrt(self.num_trails))
        ub = np.squeeze(mean + self.z * std / np.sqrt(self.num_trails))

        if self.ydata.find('acc') != -1:
            print(f"{self.ydata}, final acc is {mean[-1]}")

        ax.plot( self.xvals[0::self.sub_sample], mean[0::self.sub_sample], label=label, color=color,linestyle=line, linewidth=1.7)
        if shadow_flag:
            ax.fill_between(self.xvals[0::self.sub_sample], lb[0::self.sub_sample], ub[0::self.sub_sample], color=color, alpha=.05)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.grid()
        ax.tick_params(axis='both', which='major', labelsize=13)

        if ax_insert is not None:
            left_end, right_end, = range_
            left_end_index = next((i for i, x in enumerate(self.xvals) if x > left_end), -1)
            right_end_index = next((i for i, x in enumerate(self.xvals) if x > right_end), -1)
            ax_insert.plot( self.xvals[left_end_index:right_end_index:self.sub_sample], mean[left_end_index:right_end_index:self.sub_sample], label=label, color=color,linestyle=line, linewidth=2)
            ax_insert.set_xscale('log')
            ax_insert.grid(True)