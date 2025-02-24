import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
from sklearn.model_selection import validation_curve
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.metrics import auc
from sklearn.metrics import precision_recall_curve, average_precision_score

# 1.可视化loss学习曲线
def view_learningLoss(knn_best, X, y):
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

# 2.可视化验证曲线
def view_validationLoss(X, y):
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

# 3. 二分类可视化ROC曲线
def roccurve(y_true, y_score):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    auc_score = roc_auc_score(y_true, y_score)
    plt.plot(fpr, tpr, label='ROC curve (AUC = %0.2f)' % auc_score)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.show()

# 4. 二分类可视化PR曲线
def prcurve(y_true, y_score):
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    average_precision = average_precision_score(y_true, y_score)

    # 绘制PR曲线
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, label=f'PR curve (AP = {average_precision:.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower right")
    plt.show()
#5. 五分类knn可视化ROC曲线
def roccurve_mulitiply(y_true, y_score):
    # 计算ROC曲线和AUC

    # 假设 y_test 是测试数据的真实标签，probabilities 是模型预测的每个类别的概率
    # probabilities 应该是一个二维数组，其中每一行对应一个样本，每一列对应一个类别的概率
    n_classes = 5  # 类别数量

    # 准备存储每个类别的TPR、FPR和AUC的列表
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # 绘制所有类别的ROC曲线
    plt.figure()
    colors = ['blue', 'red', 'green', 'cyan', 'magenta', 'yellow']
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                 label=f'Class {i} (AUC = {roc_auc[i]:.2f})')

    plt.plot([0, 1], [0, 1], 'k--', lw=2)  # 随机分类器的ROC曲线
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic for Multi-class Classifier')
    plt.legend(loc="lower right")
    plt.show()

