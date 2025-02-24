import numpy as np
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# 读取数据
# 请替换为您的数据文件路径
# folder_path = "dataset_500-800/"
# folder_path = "dataset_normalization_maxmin/"
folder_path = "dataset_normalization_mean/"
# # 类别名称和标签映射
categories = {
    'ce6_0': 1, 'ce6_2-5': 1, 'ce6_5': 1, 'ce6_10': 1, 'ce6_20': 1,
    'hpts_0': 2, 'hpts_2-5': 2, 'hpts_5': 2, 'hpts_10': 2, 'hpts_20': 2,
    'Ru_0': 0, 'Ru_2-5': 0, 'Ru_5': 0, 'Ru_10': 0, 'Ru_20': 0
}
# 读取数据并分配标签
all_data = []
for category, label in categories.items():
    data = np.loadtxt(f'{folder_path}{category}.txt')
    labeled_data = np.c_[data, label * np.ones(len(data))]
    all_data.append(labeled_data)

# 合并数据集
all_data = np.vstack(all_data)

# 拆分特征和标签
X = data[:, :-1]  # 特征
y = data[:, -1]   # 标签

# 数据预处理
# 这里可以添加数据预处理的步骤，如归一化、标准化等

# 按照标签进行分组
grouped_data = {}
for label in np.unique(y):
    grouped_data[label] = X[y == label]
    print(label)


# 训练 KMeans 获取质心
kmeans_models = {}
for label, group_data in grouped_data.items():
    kmeans = KMeans(n_clusters=1)  # 你可以根据需要选择聚类数目
    kmeans.fit(group_data)
    kmeans_models[label] = kmeans

# 从训练集中获取测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 定义KNN模型
knn = KNeighborsClassifier(n_neighbors=3)  # 可以根据需要调整K值

# 预测测试集标签
predicted_labels = []
for test_sample in X_test:
    distances = {}
    for label, kmeans_model in kmeans_models.items():
        cluster_centers = kmeans_model.cluster_centers_
        distances[label] = np.min(np.linalg.norm(cluster_centers - test_sample, axis=1))
    nearest_labels = sorted(distances, key=distances.get)[:3]  # 选择距离最近的3个质心
    predicted_label = max(set(nearest_labels), key=nearest_labels.count)  # 统计最近的3个质心中出现最多的类别
    predicted_labels.append(predicted_label)

# 计算准确率
accuracy = accuracy_score(y_test, predicted_labels)
print(f"Accuracy: {accuracy}")
