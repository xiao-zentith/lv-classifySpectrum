import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# 定义一个空数组用于存储所有文件的数据
all_data = np.empty((301, 0))  # 初始化为空数组，根据实际行数设定

# 文件路径列表
file_paths = [
    './dataset_normalization_500-800/ce6_0.txt',
    './dataset_normalization_500-800/ce6_2-5.txt',
    './dataset_normalization_500-800/ce6_5.txt',
    './dataset_normalization_500-800/ce6_10.txt',
    './dataset_normalization_500-800/ce6_20.txt',
    # 添加其他文件的路径...
]
# file_paths = [
#     './dataset_normalization_500-800/hpts_0.txt',
#     './dataset_normalization_500-800/hpts_2-5.txt',
#     './dataset_normalization_500-800/hpts_5.txt',
#     './dataset_normalization_500-800/hpts_10.txt',
#     './dataset_normalization_500-800/hpts_20.txt',
#     # 添加其他文件的路径...
# ]

# file_paths = [
#     './dataset_normalization_500-800/Ru_0.txt',
#     './dataset_normalization_500-800/Ru_2-5.txt',
#     './dataset_normalization_500-800/Ru_5.txt',
#     './dataset_normalization_500-800/Ru_10.txt',
#     './dataset_normalization_500-800/Ru_20.txt',
#     # 添加其他文件的路径...
# ]

# 遍历每个文件，加载数据并按列堆叠到 all_data 数组中
for file_path in file_paths:
    data = np.loadtxt(file_path)
    all_data = np.hstack((all_data, data[:, 1:]))

# 打印组合后的数据形状
print(all_data.shape)
# 提取光谱强度数据
spectra = all_data[:, 1:].T
print(spectra.shape)

# 使用PCA进行降维
pca = PCA(n_components=2)  # 设置要降到的维度，这里选择10维作为示例
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

# 将降维后的数据转换回原始空间
restored_data = pca.inverse_transform(reduced_data)
# 打印原始光谱数据的维度
print("原始光谱数据形状:", spectra.shape)
print(restored_data.shape)

# 绘制柱状图展示每个主成分的贡献率，并在柱状图上标注数值（以百分数形式）
plt.figure(figsize=(8, 6))
bars = plt.bar(range(1, len(explained_variance_ratio) + 1), explained_variance_ratio * 100)

# 在柱状图的顶部显示数值
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, yval + 0.5, round(yval, 2), va='bottom', ha='center', fontsize=10)

plt.xlabel('Principal Components')
plt.ylabel('Variance Ratio (%)')
plt.title('Variance Ratio of Principal Components')
plt.xticks(range(1, len(explained_variance_ratio) + 1))  # 设置横轴刻度为主成分编号
plt.ylim(0, max(explained_variance_ratio * 100) + 5)  # 设置纵轴范围，根据数据最大值设置
plt.show()

plt.figure(figsize=(8, 6))
plt.plot(range(1, len(explained_variance) + 1), cumulative_variance_ratio, marker='o', linestyle='-')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Variance Ratio')
plt.title('Cumulative Variance Ratio of Principal Components')
# 在曲线上标注具体数值
for i, value in enumerate(cumulative_variance_ratio):
    plt.text(i + 1, value, f'{value:.2f}', ha='right', va='bottom', fontsize=8)

plt.grid(True)
plt.show()

# 绘制单个样本在不同波长上的强度变化曲线
plt.figure(figsize=(8, 6))
# 选择一个样本（假设第一个样本）进行展示
num_samples = spectra.shape[0]  # 获取样本数量
for i in range(num_samples):
    plt.plot(data[:, 0], spectra[i], alpha=0.5)  # alpha设置透明度，使得重叠部分更明显

plt.xlabel('Wavelength')
plt.ylabel('Intensity')
plt.title('Spectrum of a Single Sample')
plt.show()

# 绘制单个样本在不同波长上的强度变化曲线
plt.figure(figsize=(8, 6))
# 选择一个样本（假设第一个样本）进行展示
num_samples = restored_data.shape[0]  # 获取样本数量
for i in range(num_samples):
    plt.plot(data[:, 0], restored_data[i], alpha=0.5)  # alpha设置透明度，使得重叠部分更明显

plt.xlabel('Wavelength')
plt.ylabel('Intensity')
plt.title('Spectrum of a Single Sample')
plt.show()
