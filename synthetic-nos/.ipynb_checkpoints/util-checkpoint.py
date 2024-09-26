import numpy as np 
import os


def create_sampling_time(logMaxIter, log_scale=True):
    """生成对数刻度或者正常刻度，sample_num记录metric运行的时间点"""
    num_points = int(1000)
    maxIter = int(10**logMaxIter)

    if log_scale:
        sample_num = np.geomspace(1, 10**logMaxIter, num_points, endpoint=False, dtype=int)
    else:
        sample_num = np.arange(0, maxIter, step=(maxIter)/num_points, dtype=int)  # 选取测算measurement的时间点
    return sample_num

class plot_figure(object):
    def __init__(self, algo_name, dir, sub_sample=20, log_flag_ = False, plot_steadystate = False, metric = "dist2ps", metric_index=0):
        self.sub_sample = sub_sample
        self.dir = dir
        self.algo_name = algo_name
        self.log_flag = log_flag_
        self.metric = metric
        self.metric_index = metric_index #when metric is train/test acc, we need to use this var to distinguish total acc/label 1/0 acc, order is 0,1,2
        self.steady_state = plot_steadystate

        self.num_trails, self.res, self.xvals = self.load_data(algo_name)
        self.z = 1.96/np.sqrt(self.num_trails) # 95% confidence， 1.645-90%

    
    def import_package(self):
        import matplotlib 
        import os 
        os.environ['PATH'] = '/usr/bin/pdflatex:' + os.environ['PATH']

        matplotlib.rcParams['ps.useafm'] = True
        matplotlib.rcParams['pdf.use14corefonts'] = True
        matplotlib.rcParams['text.usetex'] = False


    def load_data(self, algo_name):
        """从指定的数据文件中加载plot数据"""

        file = [f for f in os.listdir(self.dir) if algo_name in f]
        # 读取这些文件的数据
        files = [np.load(os.path.join(self.dir, f), allow_pickle=True) for f in file]
        # 读取measurement, 对于acc数据，由于存储的问题，需要特殊处理一下
        if self.metric in ['train_acc', 'test_acc']:
            tmp = []
            for index, f in enumerate(files):
                acc_data = f.item().get(self.metric)
                column_data = [data[self.metric_index] for data in acc_data]
                tmp.append(np.array(column_data))
            res = np.array(tmp)
        else:
            res = np.array([f.item().get(self.metric) for f in files])

        # 读取数据横坐标
        xvals = files[0].item().get('iter')

        if self.steady_state:
            self.G = np.array([f.item().get("bias") for f in files])
        
        message = f"""
        检查到有{len(res)}个数据文件: {file[:]} ...
        每个文件中gap list的长度有: {[len(r) for r in res]}
        """
        if self.log_flag:
            print(message)

        return len(file), res, xvals

    def plot_lines(self, ax, color, line='-', label='', plot_star = False, shadow_flag=True):
        mean = np.mean(self.res, axis = 0)
        std = np.std(self.res, axis = 0)

        lb = np.squeeze(mean - self.z * std / np.sqrt(self.num_trails))
        ub = np.squeeze(mean + self.z * std / np.sqrt(self.num_trails))

        ax.plot( self.xvals[0::self.sub_sample], mean[0::self.sub_sample], label=label, color=color,linestyle=line, linewidth=2)
        if shadow_flag:
            ax.fill_between(self.xvals[0::self.sub_sample], lb[0::self.sub_sample], ub[0::self.sub_sample], color=color, alpha=.05)
        ax.legend(loc=1, prop={'size': 14})
        if self.log_flag:
            ax.set_xscale('log')
            ax.set_yscale('log')
        if self.steady_state:
            ax.plot(self.xvals[0::self.sub_sample], np.mean(self.G) * np.ones_like(self.xvals[0::self.sub_sample]), color=color, lw=1.5, linestyle = line)
        ax.grid()

        if plot_star:
            ax.plot(self.xvals[0], mean[0], marker = '*', color = color, markerfacecolor=color,ms=15)