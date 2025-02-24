import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score

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
print(X.shape)
print(y.shape)
# 数据划分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)


# # 初始化并训练KNN模型
knn = KNeighborsClassifier(n_neighbors=5, weights='distance')

# 进行交叉验证
cv_scores = cross_val_score(knn, X_train, y_train, cv=5)  # 5折交叉验证

# 输出每折交叉验证的准确率
print("Cross-validation scores:{}".format(cv_scores))

# 计算交叉验证的平均准确率
print("Average cross-validation accuracy:{}".format(np.mean(cv_scores)))

# 使用最佳参数重新训练模型
best_k = np.argmax(cv_scores) + 1  # 找到最好的K值，索引+1是因为索引从0开始
knn_best = KNeighborsClassifier(n_neighbors=best_k)
knn_best.fit(X, y)

# 预测并评估模型
y_pred_best = knn_best.predict(X_test)
print(y_pred_best)
print(y_test)
accuracy_best = accuracy_score(y_test, y_pred_best)
print(f"Accuracy with best K={best_k}: {accuracy_best}")

# 可视化loss学习曲线
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve

train_sizes, train_scores, val_scores = learning_curve(knn_best, X, y, cv=5)
train_mean = np.mean(train_scores, axis=1)
val_mean = np.mean(val_scores, axis=1)

plt.figure(figsize=(8, 6))
plt.plot(train_sizes, train_mean, label='Training accuracy')
plt.plot(train_sizes, val_mean, label='Validation accuracy')
plt.xlabel('Training set size')
plt.ylabel('Accuracy')
plt.title('Learning Curve')
plt.legend()
plt.show()

from sklearn.model_selection import validation_curve

param_range = [3, 5, 7, 9, 11]  # 不同的K值
train_scores, val_scores = validation_curve(
    KNeighborsClassifier(), X, y, param_name="n_neighbors", param_range=param_range, cv=5)

train_mean = np.mean(train_scores, axis=1)
val_mean = np.mean(val_scores, axis=1)

plt.figure(figsize=(8, 6))
plt.plot(param_range, train_mean, label='Training accuracy')
plt.plot(param_range, val_mean, label='Validation accuracy')
plt.xlabel('K values')
plt.ylabel('Accuracy')
plt.title('Validation Curve')
plt.legend()
plt.show()




