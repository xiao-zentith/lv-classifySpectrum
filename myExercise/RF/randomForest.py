import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

# 类别名称和标签映射
categories = {
    'ce6_0': 1, 'ce6_2-5': 1, 'ce6_5': 1, 'ce6_10': 1, 'ce6_20': 1,
    'hpts_0': 2, 'hpts_2-5': 2, 'hpts_5': 2, 'hpts_10': 2, 'hpts_20': 2,
    'Ru_0': 3, 'Ru_2-5': 3, 'Ru_5': 3, 'Ru_10': 3, 'Ru_20': 3
}

# 读取数据并分配标签
all_data = []
for category, label in categories.items():
    data = np.loadtxt(f'{category}.txt')
    labeled_data = np.c_[data, label * np.ones(len(data))]
    all_data.append(labeled_data)

# 合并数据集
all_data = np.vstack(all_data)

# 分离特征和目标变量
X = all_data[:, 1:-1]  # 光谱波长作为特征
y = all_data[:, -1]   # 类别标签
print(X)
print(y)

# 3. 划分数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)


# 初始化并训练Random Forest模型
rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)  # 根据需要设置超参数
rf.fit(X_train, y_train)

# 预测并评估模型
y_pred = rf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# 使用PCA进行数据降维
pca = PCA(n_components=2)  # 选择2个主成分进行可视化
X_pca = pca.fit_transform(X)

# 可视化降维后的数据
plt.figure(figsize=(8, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis', alpha=0.7)
plt.title('PCA Visualization of Spectral Data')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.colorbar(label='Spectrum Label')
plt.show()

# 查看特征重要性
rf.fit(X, y)
importances = rf.feature_importances_

# 可视化特征重要性
plt.figure(figsize=(8, 6))
plt.bar(range(X.shape[1]), importances, align='center')
plt.title('Feature Importances')
plt.xlabel('Feature Index')
plt.ylabel('Importance Score')
plt.show()

