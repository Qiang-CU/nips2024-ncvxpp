import numpy as np
import pandas as pd
from sklearn.datasets import make_circles

# 设置样本数量和特征维度
num_samples = int(1e3)
dimension = 10

# 生成随机样本
X = np.random.uniform(-1, 1, size=(num_samples, dimension))

# 生成随机的 theta 向量
theta = np.random.uniform(-2, 2, size=(dimension,))
theta = np.random.norm(0, 1, size=(dimension,))
b = np.random.norm(0,1)

# 计算每个样本的标签
# y = np.round((np.sign(np.dot(X, theta) + b) + 1 ) / 2)
y = np.sign(np.dot(X, theta) + b)

# 将 X 和 y 合并为一个 DataFrame
data = pd.DataFrame(np.column_stack((X, y)), columns=[f'x_{i}' for i in range(dimension)] + ['y'])

# 保存数据到 CSV 文件
data.to_csv('data.csv', index=False)