import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from tensorflow.python.keras.models import load_model
from tensorflow.python.keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping
from tensorflow.python.keras.layers import Dropout
import tensorflow as tf
import utils.plot as plot
import seaborn as sns


# folder_path = "dataset_500-800/"
# folder_path = "dataset_normalization_maxmin/"
folder_path = "/home/asus515/Downloads/classifySpectrum/trainDemo/dataset_normalization_maxmin/"
# folder_path = "dataset_normalization_mean/"
categories_train = {
    'ce6_0': 1, 'ce6_2-5': 1, 'ce6_5': 1,
    'hpts_0': 2, 'hpts_2-5': 2, 'hpts_10': 2,
    'Ru_0': 0, 'Ru_2-5': 0,
    'C6_0': 3, 'C6_5': 3,
    'F_0': 4, 'F_2-5': 4, 'F_5': 4,
}
categories_val = {
    'ce6_10': 1,
    'hpts_5': 2,
    'Ru_10': 0,
    'C6_20': 3,
    'F_10': 4,
}
categories_test = {
    'ce6_20': 1,
    'hpts_20': 2,
    'Ru_20': 0,
    'C6_10': 3,
    'F_20': 4,
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
X_train = train_data[:, 1:-1]  # 光谱波长作为特征
y_train = train_data[:, -1]  # 类别标签

# 读取数据并分配标签
val_data = []
for category, label in categories_val.items():
    data = np.loadtxt(f'{folder_path}{category}.txt')
    data = data[0:301]
    data = tf.transpose(data, perm=[1, 0])
    labeled_data = np.c_[data, label * np.ones(len(data))]
    val_data.append(labeled_data[1:])

# 合并数据集
val_data = np.vstack(val_data)
print(val_data.shape)

# 分离特征和目标变量
X_val = val_data[:, 1:-1]  # 光谱波长作为特征
y_val = val_data[:, -1]  # 类别标签

# 读取数据并分配标签
test_data = []
for category, label in categories_test.items():
    data = np.loadtxt(f'{folder_path}{category}.txt')
    data = data[0:301]
    data = tf.transpose(data, perm=[1, 0])
    labeled_data = np.c_[data, label * np.ones(len(data))]
    test_data.append(labeled_data[1:])

# 合并数据集
test_data = np.vstack(test_data)
print(test_data.shape)

# 分离特征和目标变量
X_test = test_data[:, 1:-1]  # 光谱波长作为特征
y_test = test_data[:, -1]  # 类别标签

# 对类别标签进行编码
encoder = LabelEncoder()
y_train = encoder.fit_transform(y_train)
y_val = encoder.fit_transform(y_val)
y_test = encoder.fit_transform(y_test)

# 调整数据形状以适应一维 CNN
X_train = X_train.reshape(-1, X_train.shape[1], 1)
X_test = X_test.reshape(-1, X_test.shape[1], 1)
print(X_train.shape, X_test.shape)
print(X_train.shape[1])
# # # CNN 模型1
# model = Sequential()
# model.add(Conv1D(filters=32, kernel_size=3, activation='relu', input_shape=(X_train.shape[1], 1)))
# model.add(MaxPooling1D(pool_size=2))
# model.add(Flatten())
# model.add(Dense(10, activation='relu'))
# # model.add(Dense(3, activation='softmax'))
# model.add(Dense(len(encoder.classes_), activation='softmax'))
# print(len(encoder.classes_))
# # 编译模型
# model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
# CNN 模型2
model = Sequential()

model.add(Conv1D(filters=32, kernel_size=3, activation='relu', input_shape=(X_train.shape[1], 1), padding='SAME'))
model.add(MaxPooling1D(pool_size=2, padding='SAME', strides=2))

model.add(Conv1D(filters=64, kernel_size=3, activation='relu', padding='SAME'))
model.add(MaxPooling1D(pool_size=2, padding='SAME', strides=2))

model.add(Conv1D(filters=128, kernel_size=3, activation='relu', padding='SAME'))
model.add(MaxPooling1D(pool_size=1, padding='SAME', strides=2))

model.add(Flatten())

model.add(Dense(256, activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(128, activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(len(encoder.classes_), activation='softmax'))

# 编译模型
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# 定义 Early Stopping 回调函数 patience=5 表示如果连续 5 个 epoch 验证集的损失值没有改善，训练将提前停止
# 在 fit 方法中使用 Early Stopping 回调
checkpoint = ModelCheckpoint('model.h5', monitor='val_loss', save_best_only=True)
early_stop = EarlyStopping(monitor='val_loss', patience=10)
history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test), callbacks=[checkpoint, early_stop])
# history = model.fit(X_train, y_train, epochs=200, batch_size=32, validation_data=(X_test, y_test), callbacks=[checkpoint])

# 提取训练集和验证集上的损失值
train_loss = history.history['loss']
val_loss = history.history['val_loss']

# 绘制损失曲线
epochs = range(1, len(train_loss) + 1)
plt.plot(epochs, train_loss, label='Training Loss')
plt.plot(epochs, val_loss, label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

# Load best model
model = load_model('model.h5')
logits = model.predict(X_test)
probablities = tf.nn.softmax(logits)

# Evaluate model
score = model.evaluate(X_test, y_test, verbose=0)
print(score)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

plt.show()

from sklearn.preprocessing import label_binarize
# # # 三分类可视化
# # 将目标变量进行二进制编码
y_train_bin = label_binarize(y_train, classes=[0, 1, 2, 3, 4])
y_test_bin = label_binarize(y_test, classes=[0, 1, 2, 3, 4])
print(X_test.shape)
print(y_test_bin.shape)

from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score

# 预测测试集的类别
y_pred = model.predict_classes(X_test)
logits = model.predict(X_test)
probablities = tf.nn.softmax(logits)

conf_matrix = confusion_matrix(y_test, y_pred)
print(conf_matrix)

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
    f1_score_i = 2 * precision_i * sensitivity_i / (precision_i + sensitivity_i) if (precision_i + sensitivity_i) != 0 else 0

    sensitivity.append(sensitivity_i)
    specificity.append(specificity_i)
    precision.append(precision_i)
    accuracy1.append(accuracy_i)
    f1_Score.append(f1_score_i)

# 输出每个类别的敏感性和特异性
for i in range(len(sensitivity)):
    print(f"Class {i}: accuracy1 = {accuracy1[i]}, Sensitivity = {sensitivity[i]}, Specificity = {specificity[i]}, Precision = {precision[i]}")

plot.roccurve_mulitiply(y_test_bin, logits)

# 绘制雷达图
# 假设我们有五种分类的性能指标数据
categories = ['Class 1', 'Class 2', 'Class 3', 'Class 4', 'Class 5']
performance_metrics = {
    'accuracy': accuracy1,
    'specificity': specificity,
    'sensitivity': sensitivity,
    'precision': precision,
    'f1_score': f1_Score
}

# 使用Seaborn的heatmap函数绘制混淆矩阵
plt.figure(figsize=(10,7))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=True, xticklabels=categories, yticklabels=categories)
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Confusion Matrix')
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