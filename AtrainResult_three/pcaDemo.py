import numpy as np
import matplotlib.pyplot as plt

# 读取txt文件并将数据存储为一个矩阵(301,21)
from sklearn.decomposition import PCA

data = np.loadtxt('./dataset_500-800/ce6_0.txt')
# 第一列波长数据
wavelength = data[:, 0]

# 数据标准化处理axis=0表示按列计算(301,20)
data_standardized = np.copy(data)
data_standardized = (data[:, 1:] - np.mean(data[:, 1:], axis=0)) / np.std(data[:, 1:], axis=0)
# # 计算协方差矩阵(20,20) rowvar=False表示数据的每一列代表一个特征
cov_matrix = np.cov(data_standardized, rowvar=True)
print(cov_matrix)
print(cov_matrix.shape) #(301,301)

# 特征值分解
eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
idx = eigenvalues.argsort()[::-1]
sorted_eigenvalues = eigenvalues[idx]
sorted_eigenvectors = eigenvectors[:, idx]
# print(sorted_eigenvectors)
# # 选择前k个特征向量作为主成分
k = 10  # 设置降维后的维度
# top_k_eigenvectors = sorted_eigenvectors[:, :k]
#
# # 数据投影到选定的主成分上，得到降维后的数据
# reduced_data = np.dot(data_standardized, top_k_eigenvectors)

# 使用sklearn库实现PCA
pca = PCA(n_components=k)
reduced_data = pca.fit_transform(data_standardized)
print(reduced_data)
print(pca.explained_variance_ratio_)
print(pca.explained_variance_)
print(pca.n_components_)
print(sorted_eigenvalues[0])
new_data = np.column_stack((data[:, 0], reduced_data[:, 0], reduced_data[:, 1]))
np.savetxt('new_data.txt', new_data, delimiter='\t', fmt='%.8f')

# 画出原来的图 20个数据的均值+特征向量1*主成分1+特征向量2*主成分2 看下是原来值的百分之多少 ，不接近就在加主成分
# 画出原来的图
wavelength = data[:, 0]
intensity = data[:, 1]
plt.plot(wavelength, intensity)
plt.xlabel('波长')
plt.ylabel('光谱强度')
plt.show()

#
intensity = np.mean(data[:, 1:], axis=0) + sorted_eigenvalues[0] * sorted_eigenvectors[:, 0] \
            + sorted_eigenvalues[1] * sorted_eigenvectors[:, 1]
print(intensity)
plt.plot(wavelength, intensity)
plt.xlabel('波长1')
plt.ylabel('光谱强度1')
plt.show()
