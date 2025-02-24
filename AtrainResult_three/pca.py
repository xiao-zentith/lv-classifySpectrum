import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# 读取光谱数据
# data = np.loadtxt('./dataset_500-800/Ru_0.txt')
data = np.loadtxt('./dataset_normalization_maxmin/Ru_0.txt')
# 提取光谱强度数据
spectra = data[:, 1:].T
print(spectra.shape)

# 使用PCA进行降维
pca = PCA(n_components=10)  # 设置要降到的维度，这里选择10维作为示例
pca.fit(spectra)

# # 输出降维后的数据形状
reduced_data = pca.transform(spectra)
print("降维后的数据形状:", reduced_data.shape)
print(reduced_data)
print(pca.components_)
# 获取降维后的贡献值（方差解释度）
explained_variance = pca.explained_variance_
print("特征值：", explained_variance)

# # 画出贡献值的累积贡献率
explained_variance_ratio = pca.explained_variance_ratio_
print("每个主成分方差的比例：", explained_variance_ratio)
cumulative_variance_ratio = np.cumsum(explained_variance_ratio)
print("累计比例：", cumulative_variance_ratio)

plt.figure(figsize=(8, 6))
plt.plot(range(1, len(explained_variance) + 1), cumulative_variance_ratio, marker='o', linestyle='-')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Variance Ratio')
plt.title('Cumulative Variance Ratio of Principal Components')
plt.grid(True)
plt.show()

# 将降维后的数据转换回原始空间
restored_data = pca.inverse_transform(reduced_data)
# 打印原始光谱数据的维度
print("原始光谱数据形状:", spectra.shape)
print(restored_data.shape)

# 提取并可视化每列光谱强度数据
plt.figure(figsize=(14, 6))
plt.subplot(1, 2, 1)
plt.plot(data[:, 0], spectra.T)
plt.xlabel('Wavelength')
plt.ylabel('Intensity')
plt.title('Original Spectrum')

plt.subplot(1, 2, 2)
plt.plot(data[:, 0], restored_data.T)
plt.xlabel('Wavelength')
plt.ylabel('Intensity')
plt.title('Restored Spectrum')
plt.tight_layout()
plt.show()