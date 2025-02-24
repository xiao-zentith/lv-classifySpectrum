import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import plot
import matplotlib.pyplot as plt

# # 类别名称和标签映射
# categories = {
#     'ce6_0': 1, 'ce6_2-5': 1, 'ce6_5': 1, 'ce6_10': 1, 'ce6_20': 1,
#     'hpts_0': 2, 'hpts_2-5': 2, 'hpts_5': 2, 'hpts_10': 2, 'hpts_20': 2,
#     'Ru_0': 3, 'Ru_2-5': 3, 'Ru_5': 3, 'Ru_10': 3, 'Ru_20': 3
# }
# 类别名称和标签映射
categories = {
    'ce6_0': 1, 'ce6_2-5': 1, 'ce6_5': 1, 'ce6_10': 1, 'ce6_20': 1,
    # 'hpts_0': 2, 'hpts_2-5': 2, 'hpts_5': 2, 'hpts_10': 2, 'hpts_20': 2,
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
y = all_data[:, -1]  # 类别标签
print(X.shape)
print(y.shape)
# 数据划分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)

# 存储不同 K 值下的准确率
accuracies = []
# 定义 K 值的范围
k_values = range(1, 20)
for k in k_values:
    # 初始化并训练 KNN 模型
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)

    # 预测并计算准确率
    y_pred = knn.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    accuracies.append(accuracy)

    print(f"K={k}, Accuracy: {accuracy}")

# 使用最佳参数重新训练模型
best_k = k_values[np.argmax(accuracies)]  # 找到最好的K值，索引+1是因为索引从0开始
best_accuracy = max(accuracies)
print(f"Best K: {best_k}, Best Accuracy: {best_accuracy}")


knn_best = KNeighborsClassifier(n_neighbors=best_k)
knn_best.fit(X_train, y_train)

# 预测并评估模型
y_pred_best = knn_best.predict(X_test)
print(y_pred_best)
print(y_test)
accuracy_best = accuracy_score(y_test, y_pred_best)
print(f"Accuracy with best K={best_k}: {accuracy_best}")

# 可视化loss曲线
plot.view_learningLoss(knn_best, X, y)
plot.view_validationLoss(X, y)
# 可视化ROC曲线
y_pred_prob = knn_best.predict_proba(X_test)[:, 1]
plot.roccurve(y_test, y_pred_prob)
# 可视化PR曲线
plot.prcurve(y_test, y_pred_prob)
