import numpy as np 
import os


def create_sampling_time(logMaxIter, log_scale=True):
    """生成对数刻度或者正常刻度，sample_num记录metric运行的时间点"""
    num_points = int(5000)
    maxIter = int(10**logMaxIter)

    if log_scale:
        sample_num = np.geomspace(1, 10**logMaxIter, num_points, endpoint=False, dtype=int)
    else:
        sample_num = np.arange(0, maxIter, step=(maxIter)/num_points, dtype=int)  # 选取测算measurement的时间点
    return sample_num

def calculate_mean_list(A, window_size):
    n = len(A)
    B = [0] * n
    half_window = window_size // 2

    for i in range(n):
        # 计算窗口的起始和结束位置
        start = max(0, i - half_window)
        end = min(n, i + half_window + 1)

        # 获取窗口内的元素并计算均值
        window_elements = A[start:end]
        B[i] = sum(window_elements) / len(window_elements)

    return B


class plot_figure(object):
    def __init__(self, algo_name, dir, sub_sample=20, log_flag_ = False, plot_steadystate = False, metric = "dist2ps"):
        self.sub_sample = sub_sample
        self.dir = dir
        self.algo_name = algo_name
        self.log_flag = log_flag_
        self.metric = metric

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
        res = np.array([f.item().get(self.metric) for f in files])

        # 读取数据横坐标
        xvals = files[0].item().get('iter_ss')
        message = f"""
        检查到有{len(res)}个数据文件: {file[:]} ...
        每个文件中gap list的长度有: {[len(r) for r in res]}
        """
        if self.log_flag:
            print(message)

        return len(file), res, xvals

    def plot_lines(self, ax, color, line='-', label='', plot_star = False, shadow_flag=True, ax_insert=None, range_=None, error_tol=1e-4, smoothing = [False, 20], annotate=[False, 1e3, None]):
        import matplotlib 

        mean = np.mean(self.res, axis = 0)
        std = np.std(self.res, axis = 0)
        # get 1e-4 num sample accessed
        first_index = next((i for i, x in enumerate(mean) if x < error_tol), -1)
        print(f'the smallest num of samples achieving 1e-4 acc is: {self.xvals[first_index]}')

        smoothing_flag, window_size = smoothing
        if smoothing_flag:
            smooth_mean = calculate_mean_list(mean, window_size=window_size)
            ax.plot( self.xvals[0::self.sub_sample], smooth_mean[0::self.sub_sample], label=label, color=color,linestyle=line, linewidth=2)
            original_line_transparency = 0.02
            label = '_nolegend_'
        else:
            original_line_transparency = 1
            label = label
        
        annotate_flag, x_loc, y_loc = annotate
        if annotate_flag:
            smooth_mean = calculate_mean_list(mean, window_size=window_size)
            first_index = next((i for i, x in enumerate(smooth_mean) if x < error_tol), -1)
            ax.axhline(y=error_tol, linestyle='--', color='r', alpha=0.5, label='_nolegend_')
            ax.annotate(text=f'{self.xvals[first_index]}', xy=(self.xvals[first_index], error_tol), xytext=(self.xvals[first_index]-x_loc, y_loc),  
             arrowprops=dict(arrowstyle="->", facecolor='red', 
                            edgecolor=color, linewidth = 2, 
                            ))

        lb = np.squeeze(mean - self.z * std / np.sqrt(self.num_trails))
        ub = np.squeeze(mean + self.z * std / np.sqrt(self.num_trails))

        ax.plot( self.xvals[0::self.sub_sample], mean[0::self.sub_sample], label=label, color=color,linestyle=line, linewidth=2, alpha=original_line_transparency,)
        if shadow_flag:
            ax.fill_between(self.xvals[0::self.sub_sample], lb[0::self.sub_sample], ub[0::self.sub_sample], color=color, alpha=.05)
        ax.legend(loc=1, prop={'size': 14})
        if self.log_flag:
            ax.set_xscale('log')
            ax.set_yscale('log')
        ax.grid()

        ax.tick_params(axis='both', which='major', labelsize=13)

        if plot_star:
            ax.plot(self.xvals[0], mean[0], marker = '*', color = color, markerfacecolor=color,ms=15)

        if ax_insert is not None:
            left_end, right_end, lb, ub = range_
            left_end_index = next((i for i, x in enumerate(self.xvals) if x > left_end), -1)
            right_end_index = next((i for i, x in enumerate(self.xvals) if x > right_end), -1)
            ax_insert.plot( self.xvals[left_end_index:right_end_index:self.sub_sample], mean[left_end_index:right_end_index:self.sub_sample], label=label, color=color,linestyle=line, linewidth=2)
            ax_insert.set_xscale('log')
            ax_insert.grid(True)
            ax_insert.set_ylim([lb, ub])
            ax_insert.minorticks_off()
