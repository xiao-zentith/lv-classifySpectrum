import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import label_binarize
from sklearn.metrics import confusion_matrix
import plot
import functionUtils


folder_path = "dataset_normalization_mean/"
categories = {
    'ce6_2-5': 1, 'ce6_5': 1, 'ce6_10': 1, 'ce6_20': 1,
    'Ru_2-5': 0, 'Ru_5': 0, 'Ru_10': 0, 'Ru_20': 0
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

# 4. 初始化SVM模型 linear poly rbf sigmoid
# svm = SVC(kernel='linear', C=1.0, gamma='scale')
svm = SVC(kernel='sigmoid', C=1.0, gamma='scale', probability=True)
# svm = SVC(kernel='rbf', C=1.0, gamma='scale', probability=True)
# svm = SVC(kernel='poly', C=1.0, gamma='scale', probability=True)  # 选择合适的参数

# 5. 训练模型
svm.fit(X_train, y_train)

# 6. 模型预测
y_pred = svm.predict(X_test)

# 7. 评估性能
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

# 二分类可视化ROC曲线
y_pred_prob = svm.predict_proba(X_test)[:, 1]
plot.roccurve(y_test, y_pred_prob)
# 可视化PR曲线
plot.prcurve(y_test, y_pred_prob)


# 混淆矩阵
cm = confusion_matrix(y_test, y_pred)
print(cm)

# 计算混淆矩阵
cm = confusion_matrix(y_test, y_pred)

# 计算精确性、敏感性和特异性
# 对于每个类别的精确性和敏感性
precision = np.diag(cm) / np.sum(cm, axis=0)
recall = np.diag(cm) / np.sum(cm, axis=1)

# 计算特异性
num_classes = len(categories)
specificity = []
for i in range(num_classes):
    temp = np.delete(cm, i, 0)  # 删除第i行
    temp = np.delete(temp, i, 1)  # 删除第i列
    tn = np.sum(temp)  # True Negatives（真负例数）
    fp = np.sum(cm[:, i]) - cm[i, i]  # False Positives（假正例数）
    specificity_i = tn / (tn + fp)
    specificity.append(specificity_i)

# 输出结果
for i in range(num_classes):
    print(f"Class {i}: Precision = {precision[i]}, Recall = {recall[i]}, Specificity = {specificity[i]}")

