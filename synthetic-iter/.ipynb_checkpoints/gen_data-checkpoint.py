import numpy as np
import pandas as pd

# 设置样本数量和特征维度
num_samples = 1000
dimension = 20

# 生成随机样本
X = np.random.uniform(-5, 5, size=(num_samples, dimension))

# 生成随机的 theta 向量
theta = np.random.uniform(-1, 1, size=(dimension,))
b = np.random.uniform(-1, 1)

# 计算每个样本的标签
y = np.round((np.sign(np.dot(X, theta) + b) + 1 ) / 2)

# 将 X 和 y 合并为一个 DataFrame
data = pd.DataFrame(np.column_stack((X, y)), columns=[f'x_{i}' for i in range(dimension)] + ['y'])

# 保存数据到 CSV 文件
data.to_csv('data.csv', index=False)
