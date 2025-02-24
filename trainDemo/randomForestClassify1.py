import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import label_binarize
from sklearn.metrics import confusion_matrix
# import functionUtils
import utils.plot as plot
import tensorflow as tf
import seaborn as sns

from trainDemo.metricPlot import  plot_confusion_matrix

# folder_path = "dataset_500-800/"
# folder_path = "dataset_normalization_maxmin/"
# folder_path = "dataset_normalization_mean/"
folder_path = "/home/asus515/Downloads/classifySpectrum/trainDemo/dataset_normalization_maxmin/"
# # 类别名称和标签映射
# 类别名称和标签映射
categories_train = {
    'ce6_0': 0, 'ce6_2-5': 0, 'ce6_20': 0, 'ce6_5': 0, 'ce6_10': 0,
    'hpts_0': 1, 'hpts_2-5': 1,'hpts_20': 1, 'hpts_10': 1, 'hpts_5': 1,
    'Ru_0': 2, 'Ru_2-5': 2,'Ru_20': 2, 'Ru_5': 2, 'Ru_10': 2,
    'C6_0': 3, 'C6_5': 3, 'C6_10': 3, 'C6_20': 3, 'C6_2-5': 3,
    'F_0': 4, 'F_2-5': 4, 'F_20': 4, 'F_5': 4, 'F_10': 4,
}
# 读取数据并分配标签
train_data = []
for category, label in categories_train.items():
    data = np.loadtxt(f'{folder_path}{category}.txt')
    data = data[0:301]
    data = tf.transpose(data, perm=[1, 0])
    labeled_data = np.c_[data, label * np.ones(len(data))]
    train_data.append(labeled_data[1:])

# 合并数据集
train_data = np.vstack(train_data)
print(train_data.shape)

# 分离特征和目标变量
X = train_data[:, 1:-1]  # 光谱波长作为特征
y = train_data[:, -1]  # 类别标签

# 添加噪声
noise_normal = np.random.normal(0, 0.12, (481, 300))  # 正态分布噪声
noise_uniform = np.random.uniform(-0.05, 0.05, (481, 300))  # 均匀分布噪声

# 将不同类型的噪声相加
total_noise = noise_normal + noise_uniform

# 将噪声添加到光谱曲线上
X = X + total_noise

# 数据划分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, shuffle=True)

#标准化应该就是归一化
# X_train, X_test = functionUtils.standard(X_train, X_test)
# 初始化PCA并拟合训练集
# X_train, X_test = functionUtils.pca(2, X_train, X_test)

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
pca = PCA(n_components=5)  # 选择2个主成分进行可视化
X_pca = pca.fit_transform(X_train)

# 可视化降维后的数据
plt.figure(figsize=(8, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y_train, cmap='viridis', alpha=0.7)
plt.title('PCA Visualization of Spectral Data')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.colorbar(label='Spectrum Label')
plt.show()

# 查看特征重要性
rf.fit(X_train, y_train)
importances = rf.feature_importances_

# 可视化特征重要性
plt.figure(figsize=(8, 6))
plt.bar(range(X_train.shape[1]), importances, align='center')
plt.title('Feature Importances')
plt.xlabel('Feature Index')
plt.ylabel('Importance Score')
plt.show()

# # # 三分类可视化
# # 将目标变量进行二进制编码
y_train_bin = label_binarize(y_train, classes=[0, 1, 2, 3, 4])
y_test_bin = label_binarize(y_test, classes=[0, 1, 2, 3, 4])
print(X_test.shape)
print(y_test_bin.shape)

# #三分类
# 将KNN与OneVsRestClassifier结合
classifier = OneVsRestClassifier(rf)
# 训练模型
classifier.fit(X_train, y_train_bin)
# 获取决策得分的概率
y_score = classifier.predict_proba(X_test)
plot.roccurve_mulitiply(y_test_bin, y_score)

conf_matrix = confusion_matrix(y_test, y_pred)

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
f1_Score = []
for i in range(conf_matrix.shape[0]):
    TP = conf_matrix[i, i]
    FN = np.sum(conf_matrix[i, :]) - TP
    FP = np.sum(conf_matrix[:, i]) - TP
    TN = np.sum(conf_matrix) - TP - FN - FP

    sensitivity_i = TP / (TP + FN) if (TP + FN) != 0 else 0
    specificity_i = TN / (TN + FP) if (TN + FP) != 0 else 0
    precision_i = TP / (TP + FP) if (TP + FP) != 0 else 0
    accuracy_i = (TP+TN) / (TP + FN + TN + FP) if (TP + FN + TN + FP) != 0 else 0
    f1_Score_i = 2 * precision_i * sensitivity_i / (precision_i + sensitivity_i) if (precision_i + sensitivity_i) != 0 else 0

    sensitivity.append(sensitivity_i)
    specificity.append(specificity_i)
    precision.append(precision_i)
    accuracy1.append(accuracy_i)
    f1_Score.append(f1_Score_i)


# 输出每个类别的敏感性和特异性
for i in range(len(sensitivity)):
    print(f"Class {i}: accuracy1 = {accuracy1[i]}, Sensitivity = {sensitivity[i]}, Specificity = {specificity[i]}, Precision = {precision[i]}")

# 绘制雷达图
# 假设我们有五种分类的性能指标数据
categories = ['Ce6', 'Hpts', 'Ru', 'C6', 'FITC']
performance_metrics = {
    'accuracy': accuracy1,
    'specificity': specificity,
    'sensitivity': sensitivity,
    'precision': precision,
    'f1_score': f1_Score
}

# 计算每行的准确率 (对角线元素除以行总和)
row_accuracy = np.diag(conf_matrix) / conf_matrix.sum(axis=1)

# 计算每列的比重 (列总和除以总和)
column_proportions = conf_matrix.sum(axis=0) / conf_matrix.sum()

# 扩大混淆矩阵
cm_expanded = np.hstack((conf_matrix, row_accuracy.reshape(-1, 1)))
cm_expanded = np.vstack((cm_expanded, np.append(column_proportions, 0)))

# 添加最后一行的最后一列的值为0
cm_expanded[-1, -1] = (row_accuracy * column_proportions).sum()


# 使用Seaborn的heatmap函数绘制混淆矩阵
plt.figure(figsize=(10,7))
sns.set(font_scale=1.5)
sns.heatmap(cm_expanded, annot=True, cmap='Blues', cbar=False, xticklabels=categories + ['Accuracy'], yticklabels=categories + ['Proportion'])
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('RF Confusion Matrix')
plt.show()

# 计算每个分类的雷达图顶点数
num_vars = len(performance_metrics['accuracy'])

# 制作雷达图
angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
# 将角度转换为弧度
angles += angles[:1]

fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))

# 绘制每个分类的雷达图边
colors = ['b', 'g', 'r', 'c', 'm']  # 五种不同的颜色
for idx, category in enumerate(categories):
    values = [performance_metrics[metric][idx] for metric in performance_metrics]
    values += values[:1]  # 闭合图形
    ax.plot(angles, values, color=colors[idx], linewidth=2, linestyle='solid', marker='o', markersize=5, label=category)

# 设置雷达图的刻度和标签
# ax.set_yticklabels([])  # 不显示y轴刻度
ax.set_xticks(angles[:-1])
ax.set_xticklabels(['Accuracy', 'Specificity', 'Sensitivity', 'Precision', 'F1-Score'])

# 设置雷达图的标题
ax.set_title("Performance Metrics for Five Classes", size=15, color='black', y=1.1)

# 显示图例
plt.legend(loc='upper right', bbox_to_anchor=(1.1, 1.05))

# 显示图表
plt.show()
