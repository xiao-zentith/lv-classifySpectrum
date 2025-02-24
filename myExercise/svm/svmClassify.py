import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
import plot

# 类别名称和标签映射
categories = {
    'ce6_0': 1, 'ce6_2-5': 1, 'ce6_5': 1, 'ce6_10': 1, 'ce6_20': 1,
    # 'hpts_0': 0, 'hpts_2-5': 0, 'hpts_5': 0, 'hpts_10': 0, 'hpts_20': 0
    'Ru_0': 0, 'Ru_2-5': 0, 'Ru_5': 0, 'Ru_10': 0, 'Ru_20': 0
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


# 4. 初始化SVM模型 linear poly rbf sigmoid
# svm = SVC(kernel='linear', C=1.0, gamma='scale')
svm = SVC(kernel='sigmoid', C=1.0, gamma='scale')  # 选择合适的参数

# 5. 训练模型
svm.fit(X_train, y_train)

# 6. 模型预测
y_pred = svm.predict(X_test)

# 7. 评估性能
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

# 可视化
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# 使用PCA降维至2维
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# 初始化并训练SVM模型
svm = SVC(kernel='sigmoid', C=1.0, gamma='scale')
svm.fit(X_pca, y)

# 取数据范围的最小值和最大值
x_min, x_max = X_pca[:, 0].min() - 1, X_pca[:, 0].max() + 1
y_min, y_max = X_pca[:, 1].min() - 1, X_pca[:, 1].max() + 1

# 创建网格以评估模型
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
Z = svm.decision_function(np.c_[xx.ravel(), yy.ravel()])
print(Z.shape)
print(xx.shape)
Z = Z.reshape(xx.shape)

# 画出决策边界和间隔
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis', s=50)
plt.contourf(xx, yy, Z, alpha=0.5, cmap='coolwarm')
plt.show()

# y_pred_prob = svm.predict_proba(X_test)[:, 1]
plot.roccurve(y_test, y_pred)
# 可视化PR曲线
plot.prcurve(y_test, y_pred)
