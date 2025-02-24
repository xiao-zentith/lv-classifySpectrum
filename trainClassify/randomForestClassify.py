import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import label_binarize
from sklearn.metrics import confusion_matrix
import functionUtils
import plot

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

# 分离特征和目标变量
X = all_data[:, 1:-1]  # 光谱波长作为特征
y = all_data[:, -1]  # 类别标签
print(X)
print(y)
# 数据划分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)
#标准化应该就是归一化
# X_train, X_test = functionUtils.standard(X_train, X_test)
# 初始化PCA并拟合训练集
# X_train, X_test = functionUtils.pca(2, X_train, X_test)

# 初始化并训练Random Forest模型
rf = RandomForestClassifier(n_estimators=400, max_depth=10, random_state=42)  # 根据需要设置超参数
rf.fit(X_train, y_train)

# 预测并评估模型
y_pred = rf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# 使用PCA进行数据降维
pca = PCA(n_components=3)  # 选择2个主成分进行可视化
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

# # 三分类可视化
# # 将目标变量进行二进制编码
y_train_bin = label_binarize(y_train, classes=[0, 1, 2])
y_test_bin = label_binarize(y_test, classes=[0, 1, 2])
print(X_test.shape)
print(y_test_bin.shape)
#
# # 二分类可视化ROC曲线
# y_pred_prob = rf.predict_proba(X_test)[:, 1]
# plot.roccurve(y_test, y_pred_prob)
# # 可视化PR曲线
# plot.prcurve(y_test, y_pred_prob)

# #三分类
# 将KNN与OneVsRestClassifier结合
classifier = OneVsRestClassifier(rf)
# 训练模型
classifier.fit(X_train, y_train_bin)
# 获取决策得分的概率
y_score = classifier.predict_proba(X_test)
plot.roccurve_mulitiply(y_test_bin, y_score)

# 混淆矩阵
cm = confusion_matrix(y_test, y_pred)
print(cm)