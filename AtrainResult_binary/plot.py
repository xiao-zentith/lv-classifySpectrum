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
#5. 三分类knn可视化ROC曲线
def roccurve_mulitiply(y_true, y_score):
    # 计算ROC曲线和AUC
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(3):  # 循环处理每个类别
        fpr[i], tpr[i], _ = roc_curve(y_true[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    class_labels = {
        1: 'ce6',
        2: 'hpts',
        0: 'Ru'
    }

    # 绘制多分类ROC曲线
    plt.figure(figsize=(8, 6))
    for i in range(3):
        plt.plot(fpr[i], tpr[i], label=f'{class_labels [i]} (AUC = {roc_auc[i]:.2f})')

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('CNN---Multiclass ROC Curve')
    plt.legend(loc="lower right")
    plt.show()

