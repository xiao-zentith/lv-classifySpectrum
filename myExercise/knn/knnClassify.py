import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# 文件夹路径
folder_path = '/home/asus515/Downloads/classifySpectrum/myExercise/knn'

# 读取数据并准备数据集
X_data = []
y_data = []

for substance in ['ce6', 'hpts', 'Ru']:
    for label in ['0', '2-5', '5', '10', '20']:
        file_name = f"{substance}_{label}.txt"
        if file_name in os.listdir(folder_path):
            with open(os.path.join(folder_path, file_name), 'r') as file:
                lines = file.readlines()
                # wavelengths = []
                intensities = []
                for line in lines:
                    data = line.split()
                    # wavelengths.append(float(data[0]))
                    intensities.append([float(val) for val in data[1:]])

                # wavelengths = np.array(wavelengths)
                intensities = np.array(intensities)
                X_data.append(intensities)
                y_data.append(substance)
for i, d in enumerate(data):
    print(f"Data {i+1} length: {len(d)}")

# 转换数据为NumPy数组并进行归一化处理
X_data = np.array(X_data)
X_data = (X_data - np.min(X_data)) / (np.max(X_data) - np.min(X_data))  # 归一化

# 对标签进行编码
label_map = {'ce6': 0, 'hpts': 1, 'Ru': 2}
y_data = [label_map[label] for label in y_data]
print(X_data.shape)
print(y_data)

# 划分数据集为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X_data, np.array(y_data), test_size=0.2, random_state=42)

X_data_2d = X_train.reshape(X_train.shape[0], -1)
y_data_2d = y_train.reshape(y_train.shape[0], -1)
X_test_2d = X_test.reshape(X_test.shape[0], -1)

# 创建并训练 KNN 模型
knn = KNeighborsClassifier(n_neighbors=3)  # 设置 K 值为 3
knn.fit(X_data_2d, y_data_2d)
print(X_test.shape)
# 预测测试集
y_pred = knn.predict(X_test_2d)

# 计算准确率
accuracy = np.mean(y_pred == y_test)
print(f"Accuracy: {accuracy}")

# 可视化模型结果
# 可以使用降维算法（如PCA或t-SNE）将高维数据可视化到二维/三维空间

# 如果想在二维空间可视化，可以使用 PCA
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_data_2d)

plt.figure(figsize=(8, 6))
for label in np.unique(y_data_2d):
    mask = (y_data_2d == label)
    plt.scatter(X_pca[mask, 0], X_pca[mask, 1], label=label)
plt.legend()
plt.title('PCA Visualization of Spectral Data')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.show()
