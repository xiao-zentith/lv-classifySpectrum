import numpy as np

# 定义一个空数组，用于存储所有样品的光谱数据
data = np.zeros((20, 20))

# 循环读取每个数据文件，并将20个光谱数据添加到data数组中
for i in range(1, 21):
    filename = f"{i}.txt"
    with open(filename, 'r') as file:
        # 读取每个文件中的所有行
        lines = [line.split() for line in file]
        # 获取每个样品的光谱数据，转成一维数组
        spec_data = [float(line[j]) for line in lines for j in range(1)]
        # 将光谱数据添加到data数组中
        data[i-1,:] = spec_data

from sklearn.decomposition import PCA

# 使用PCA对数据矩阵进行降维处理
pca = PCA(n_components=2)
pca.fit(data)
reduced_data = pca.transform(data)

# import matplotlib.pyplot as plt
#
# # 将降维后的数据在二维平面绘制散点图
# ce6_data = reduced_data[:100,:]
# hpts_data = reduced_data[100:200,:]
# ru_data = reduced_data[200:,:]
#
# plt.scatter(ce6_data[:,0], ce6_data[:,1], c='red', alpha=0.5, label='ce6')
# plt.scatter(hpts_data[:,0], hpts_data[:,1], c='blue', alpha=0.5, label='hpts')
# plt.scatter(ru_data[:,0], ru_data[:,1], c='green', alpha=0.5, label='ru')
# plt.legend()
# plt.show()