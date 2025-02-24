import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# 示例混淆矩阵
cm = np.array([[25,  2,  1,  0,  2],
               [ 3, 18,  2,  1,  6],
               [ 0,  2, 15,  3,  0],
               [ 0,  6,  2, 20,  2],
               [ 1,  2,  1,  0, 24]])

# 类别标签
class_labels = ['ce6', 'hpts', 'Ru', 'C6', 'F']

# 计算每行的准确率 (对角线元素除以行总和)
row_accuracy = np.diag(cm) / cm.sum(axis=1)

# 计算每列的比重 (列总和除以总和)
column_proportions = cm.sum(axis=0) / cm.sum()

# 扩大混淆矩阵
cm_expanded = np.hstack((cm, column_proportions.reshape(-1, 1)))
cm_expanded = np.vstack((cm_expanded, np.append(row_accuracy, 0)))

# 添加最后一行的最后一列的值为0
cm_expanded[-1, -1] = (row_accuracy * column_proportions).sum()

# # 更新类别标签，加入准确率和比重的标签
# class_labels += ['Accuracy']
# row_accuracy_labels = [f"Accuracy: {acc:.2%}" for acc in row_accuracy]
# class_labels = class_labels + row_accuracy_labels

# 绘制混淆矩阵
plt.figure(figsize=(10, 7))
sns.set(font_scale=1.5)
sns.heatmap(cm_expanded, annot=True, cmap="Blues", xticklabels=class_labels,
            yticklabels=class_labels, cbar=False)  # 不显示颜色比例尺


# 设置左侧和上方的类别标签
plt.yticks(np.arange(len(class_labels) + 1), class_labels + ['Accuracy'], va="center", rotation = 0)
plt.xticks(np.arange(len(class_labels) + 1), class_labels + ['Proportion'], ha="center", rotation = 0)

# 设置图表标题和轴标签
plt.title("KNN Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")

# 显示图表
plt.show()