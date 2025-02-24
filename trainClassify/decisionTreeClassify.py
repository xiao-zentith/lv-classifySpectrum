import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import learning_curve
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import label_binarize
from sklearn.metrics import confusion_matrix
import plot
import functionUtils

folder_path = "dataset_500-800/"
# folder_path = "dataset_normalization_maxmin/"
# folder_path = "dataset_normalization_mean/"
# # 类别名称和标签映射
# categories = {
#     'ce6_0': 1, 'ce6_2-5': 1, 'ce6_5': 1, 'ce6_10': 1, 'ce6_20': 1,
#     'hpts_0': 2, 'hpts_2-5': 2, 'hpts_5': 2, 'hpts_10': 2, 'hpts_20': 2,
#     'Ru_0': 0, 'Ru_2-5': 0, 'Ru_5': 0, 'Ru_10': 0, 'Ru_20': 0
# }
categories = {
    'ce6_20': 1, 'ce6_5': 1, 'ce6_10': 1, 'ce6_20': 1,
    # 'hpts_0': 2, 'hpts_2-5': 2, 'hpts_5': 2, 'hpts_10': 2, 'hpts_20': 2,
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

# 初始化并训练决策树模型
decision_tree = DecisionTreeClassifier()
# 定义要调整的参数网格
param_grid = {
    'max_depth': [3, 5, 7],  # 搜索树的最大深度
    'min_samples_split': [2, 5, 10],  # 搜索节点分裂所需的最小样本数
    'min_samples_leaf': [1, 2, 4]  # 搜索叶节点所需的最小样本数
}
# 初始化网格搜索
grid_search = GridSearchCV(decision_tree, param_grid, cv=5)

# 进行网格搜索
grid_search.fit(X_train, y_train)  # X是特征，y是目标变量
# 输出最佳参数
print("Best parameters:", grid_search.best_params_)

# 使用最佳参数重新训练模型
best_decision_tree = grid_search.best_estimator_

best_decision_tree.fit(X_train, y_train)
# 进行交叉验证
cv_scores = cross_val_score(best_decision_tree, X_train, y_train, cv=5)  # X是特征，y是目标变量，cv是交叉验证折数

# 输出每折交叉验证的准确率
print("Cross-validation scores:", cv_scores)

# 计算交叉验证的平均准确率
print("Average cross-validation accuracy:", np.mean(cv_scores))

# 预测并评估模型
y_pred = best_decision_tree.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

# 绘制学习曲线
train_sizes, train_scores, test_scores = learning_curve(
    best_decision_tree, X, y, cv=5, train_sizes=np.linspace(0.1, 1.0, 10)
)

# 计算训练集和验证集的平均准确率
train_mean = np.mean(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)

# 绘制训练集得分曲线
plt.figure(figsize=(8, 6))
plt.plot(train_sizes, train_mean, label='Training accuracy')
plt.xlabel('Number of training samples')
plt.ylabel('Accuracy')
plt.title('Training Curve')
plt.legend()
plt.show()

# 绘制验证集得分曲线
plt.figure(figsize=(8, 6))
plt.plot(train_sizes, test_mean, label='Validation accuracy')
plt.xlabel('Number of training samples')
plt.ylabel('Accuracy')
plt.title('Validation Curve')
plt.legend()
plt.show()

# 获取预测概率值
y_score = best_decision_tree.predict_proba(X_test)[:, 1]

# # 二分类可视化ROC曲线
# y_pred_prob = best_decision_tree.predict_proba(X_test)[:, 1]
# plot.roccurve(y_test, y_pred_prob)
# # 可视化PR曲线
# plot.prcurve(y_test, y_pred_prob)

# # 三分类可视化
# # 将目标变量进行二进制编码
y_train_bin = label_binarize(y_train, classes=[0, 1, 2])
y_test_bin = label_binarize(y_test, classes=[0, 1, 2])
# 将KNN与OneVsRestClassifier结合
classifier = OneVsRestClassifier(best_decision_tree)
# 训练模型
classifier.fit(X_train, y_train_bin)
# 获取决策得分的概率
y_score = classifier.predict_proba(X_test)
plot.roccurve_mulitiply(y_test_bin, y_score)

# 混淆矩阵
cm = confusion_matrix(y_test, y_pred)
print(cm)



