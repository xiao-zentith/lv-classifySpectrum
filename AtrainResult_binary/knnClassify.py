import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import plot
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import label_binarize
from sklearn.metrics import confusion_matrix
import functionUtils
import matplotlib.pyplot as plt

folder_path = "dataset_normalization_mean/"
# # 类别名称和标签映射
categories = {
    'Ru_2-5': 0, 'Ru_5': 0, 'Ru_10': 0, 'Ru_20': 0,
    'ce6_2-5': 1, 'ce6_5': 1, 'ce6_10': 1, 'ce6_20': 1
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
# 标准化应该就是归一化
# X_train, X_test = functionUtils.standard(X_train, X_test)
# 初始化PCA并拟合训练集
# X_train, X_test = functionUtils.pca(15, X_train, X_test)

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
accuracy_best = accuracy_score(y_test, y_pred_best)
print(f"Accuracy with best K={best_k}: {accuracy_best}")

# 二分类可视化ROC曲线
y_pred_prob = knn_best.predict_proba(X_test)[:, 1]
plot.roccurve(y_test, y_pred_prob)
# 可视化PR曲线
plot.prcurve(y_test, y_pred_prob)

conf_matrix = confusion_matrix(y_test, y_pred_best)
print("confusion Matrix:", conf_matrix)
precision = conf_matrix[1, 1] / (conf_matrix[1, 1] + conf_matrix[0, 1])
print("presicion", precision)
recall = conf_matrix[1, 1] / (conf_matrix[1, 1] + conf_matrix[1, 0])
print("recall", recall)
specificity = conf_matrix[0, 0] / (conf_matrix[0, 0] + conf_matrix[0, 1])
print("specificity", specificity)

# 提取混淆矩阵中的 TP, TN, FP, FN
TP = conf_matrix.diagonal()  # 对角线上的值即为每个类别的 TP
FP = conf_matrix.sum(axis=0) - TP  # 列求和减去对角线上的值即为每个类别的 FP
FN = conf_matrix.sum(axis=1) - TP  # 行求和减去对角线上的值即为每个类别的 FN
TN = conf_matrix.sum() - (TP + FP + FN)  # 总数减去其他三个值即为每个类别的 TN

# 输出结果
print("True Positives (TP):", TP)
print("True Negatives (TN):", TN)
print("False Positives (FP):", FP)
print("False Negatives (FN):", FN)

# 假设 conf_matrix 为混淆矩阵
# 计算敏感性和特异性
sensitivity = []
specificity = []
precision = []
accuracy1 = []
for i in range(conf_matrix.shape[0]):
    TP = conf_matrix[i, i]
    FN = np.sum(conf_matrix[i, :]) - TP
    FP = np.sum(conf_matrix[:, i]) - TP
    TN = np.sum(conf_matrix) - TP - FN - FP

    sensitivity_i = TP / (TP + FN) if (TP + FN) != 0 else 0
    specificity_i = TN / (TN + FP) if (TN + FP) != 0 else 0
    precision_i = TP / (TP + FP) if (TP + FP) != 0 else 0
    accuracy_i = (TP+TN) / (TP + FN + TN + FP) if (TP + FN + TN + FP) != 0 else 0

    sensitivity.append(sensitivity_i)
    specificity.append(specificity_i)
    precision.append(precision_i)
    accuracy1.append(accuracy_i)

# 输出每个类别的敏感性和特异性
for i in range(len(sensitivity)):
    print(f"Class {i}: accuracy1 = {accuracy1[i]}, Sensitivity = {sensitivity[i]}, Specificity = {specificity[i]}, Precision = {precision[i]}")

from sklearn.metrics import roc_curve, auc

# 使用模型进行预测，获取预测概率
y_probs = knn_best.predict_proba(X_test)

# 计算 ROC 曲线的参数
fpr, tpr, thresholds = roc_curve(y_test, y_probs[:, 1])  # 这里假设 y_probs 的第一列是正例的概率

# 计算 AUC
roc_auc = auc(fpr, tpr)

# 绘制 ROC 曲线
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()

