import torch
import numpy as np
import matplotlib.pyplot as plt

# 假设我们有一个五分类模型的输出概率，以及对应的真实标签
# 真实标签是一个one-hot编码的张量，例如：torch.tensor([[1, 0, 0, 0, 0], ...])
# 模型输出概率是一个张量，例如：torch.tensor([[0.1, 0.2, 0.3, 0.1, 0.3], ...])

def calculate_roc_curve(y_true, y_scores, num_classes):
    fpr = np.zeros((num_classes, 1))
    tpr = np.zeros((num_classes, 1))
    thresholds = np.linspace(0, 1, 1000)  # 生成一系列阈值

    for i in range(num_classes):
        # 计算当前类别的TPR和FPR
        y_true_class = (y_true[:, i] == 1).astype(np.float32)  # 转换为布尔值然后转换为浮点数
        y_scores_class = y_scores[:, i]

        tp_by_thresh = np.zeros_like(thresholds)
        fp_by_thresh = np.zeros_like(thresholds)

        # 对每个阈值计算TP和FP
        for j, thresh in enumerate(thresholds):
            tp_by_thresh[j] = np.sum((y_scores_class > thresh) & (y_true_class == 1))
            fp_by_thresh[j] = np.sum((y_scores_class > thresh) & (y_true_class == 0))

        tp_cumsum = np.cumsum(tp_by_thresh)
        fp_cumsum = np.cumsum(fp_by_thresh)
        recall = tp_cumsum / np.sum(y_true_class)
        precision = tp_cumsum / (tp_cumsum + fp_cumsum)

        # 计算FPR和TPR
        fpr[i] = 1 - precision
        tpr[i] = recall

    return fpr, tpr

def roccurve_mulitiply(y_true, y_scores):
    # 计算ROC曲线
    fpr, tpr = calculate_roc_curve(y_true, y_scores, 5)

    # 绘图
    plt.figure(figsize=(10, 8))
    for i in range(5):
        plt.plot(fpr[i], tpr[i], label=f'Class {i+1}')

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic for multi-class data')
    plt.legend(loc="lower right")
    plt.show()