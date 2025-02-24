import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score


# 读取三个类别的光谱数据文件
data_class_1 = np.loadtxt('ce6_0.txt')  # 第一个类别的数据文件
data_class_2 = np.loadtxt('hpts_0.txt')  # 第二个类别的数据文件
data_class_3 = np.loadtxt('Ru_0.txt')  # 第三个类别的数据文件

# 分配类别标签
data_class_1 = np.c_[data_class_1, np.ones(len(data_class_1))]  # 第一个类别标签为1
data_class_2 = np.c_[data_class_2, 2 * np.ones(len(data_class_2))]  # 第二个类别标签为2
data_class_3 = np.c_[data_class_3, 3 * np.ones(len(data_class_3))]  # 第三个类别标签为3

# 合并数据集
all_data = np.vstack((data_class_1, data_class_2, data_class_3))

# 分离特征和目标变量
X = all_data[:, 1:-1]  # 光谱波长作为特征
y = all_data[:, -1]   # 类别标签
print(X.shape)
print(y)

# 数据划分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # 初始化并训练KNN模型
knn = KNeighborsClassifier(n_neighbors=5, weights='distance')

# 进行交叉验证
cv_scores = cross_val_score(knn, X, y, cv=5)  # 5折交叉验证

# 输出每折交叉验证的准确率
print("Cross-validation scores:", cv_scores)

# 计算交叉验证的平均准确率
print("Average cross-validation accuracy:", np.mean(cv_scores))

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




# 读取所有数据并整合
ce_data = []
hpts_data = []
Ru_data = []


# # # 读取三个类别的光谱数据文件
# ce_0_data = np.loadtxt('ce6_0.txt')  # 第一个类别的数据文件
# ce_2_5_data = np.loadtxt('ce6_2-5.txt')
# ce_5_data = np.loadtxt('ce6_5.txt')
# ce_10_data = np.loadtxt('ce6_10.txt')
# ce_20_data = np.loadtxt('ce6_20.txt')
# hpts_0_data = np.loadtxt('hpts_0.txt')  # 第二个类别的数据文件
# hpts_2_5_data = np.loadtxt('hpts_2-5.txt')
# hpts_5_data = np.loadtxt('hpts_5.txt')
# hpts_10_data = np.loadtxt('hpts_10.txt')
# hpts_20_data = np.loadtxt('hpts_20.txt')
# Ru_0_data = np.loadtxt('Ru_0.txt')  # 第三个类别的数据文件
# Ru_2_5_data = np.loadtxt('Ru_2-5.txt')
# Ru_5_data = np.loadtxt('Ru_5.txt')
# Ru_10_data = np.loadtxt('Ru_10.txt')
# Ru_20_data = np.loadtxt('Ru_20.txt')
#
#
# # # 分配类别标签
# ce_0_data = np.c_[ce_0_data, np.ones(len(ce_0_data))]  # 第一个类别标签为1
# ce_2_5_data = np.c_[ce_2_5_data, np.ones(len(ce_2_5_data))]
# ce_5_data = np.c_[ce_5_data, np.ones(len(ce_5_data))]
# hpts_0_data = np.c_[hpts_0_data, 2 * np.ones(len(hpts_0_data))]  # 第二个类别标签为2
# hpts_2_5_data = np.c_[hpts_2_5_data, 2 * np.ones(len(hpts_2_5_data))]
# hpts_5_data = np.c_[hpts_5_data, 2 * np.ones(len(hpts_5_data))]
# hpts_10_data = np.c_[hpts_10_data, 2 * np.ones(len(hpts_10_data))]
# hpts_20_data = np.c_[hpts_20_data, 2 * np.ones(len(hpts_20_data))]
# Ru_0_data = np.c_[Ru_0_data, 3 * np.ones(len(Ru_0_data))]  # 第三个类别标签为3
# Ru_2_5_data = np.c_[Ru_2_5_data, 3 * np.ones(len(Ru_2_5_data))]
# Ru_5_data = np.c_[Ru_5_data, 3 * np.ones(len(Ru_5_data))]
# Ru_10_data = np.c_[Ru_10_data, 3 * np.ones(len(Ru_10_data))]
# Ru_20_data = np.c_[Ru_20_data, 3 * np.ones(len(Ru_20_data))]
#
# # 合并数据集
# all_data = np.vstack((ce_0_data, ce_2_5_data, ce_5_data, hpts_0_data, hpts_2_5_data, hpts_5_data,  hpts_10_data, hpts_20_data, Ru_0_data, Ru_2_5_data, Ru_5_data, Ru_10_data, Ru_20_data))
