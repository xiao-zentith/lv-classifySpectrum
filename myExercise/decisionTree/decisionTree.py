import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import label_binarize

# 类别名称和标签映射
categories = {
    'ce6_0': 1, 'ce6_2-5': 1, 'ce6_5': 1, 'ce6_10': 1, 'ce6_20': 1,
    'hpts_0': 2, 'hpts_2-5': 2, 'hpts_5': 2, 'hpts_10': 2, 'hpts_20': 2
    # 'Ru_0': 3, 'Ru_2-5': 3, 'Ru_5': 3, 'Ru_10': 3, 'Ru_20': 3
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
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)

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
grid_search.fit(X, y)  # X是特征，y是目标变量
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

# 将多类别标签转换为二进制标签
y_binary = label_binarize(y_test, classes=[1, 2])  # 将类别1和2转换为0和1

# 获取预测概率值
y_scores = best_decision_tree.decision_function(X_test)

# 计算Precision和Recall
precision, recall, thresholds = precision_recall_curve(y_binary.ravel(), y_scores.ravel())

# 计算平均Precision
average_precision = average_precision_score(y_binary.ravel(), y_scores.ravel())

# 绘制Precision-Recall曲线
plt.figure(figsize=(8, 6))
plt.step(recall, precision, where='post', label=f'AP={average_precision}')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend()
plt.show()