import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# 文件夹路径
folder_path = '/home/asus515/Downloads/classifySpectrum/myExercise/cnn'

# 读取数据并准备数据集
X_data = []
y_data = []

# 针对每个文件
for substance in ['ce6', 'hpts', 'Ru']:
    for label in ['0', '2-5', '5', '10', '20']:
        file_name = f"{substance}_{label}.txt"
        if file_name in os.listdir(folder_path):
            with open(os.path.join(folder_path, file_name), 'r') as file:
                lines = file.readlines()
                intensities_per_file = []  # 存储每个文件的数据

                # 逐列读取文件中的数据
                for line in lines:
                    data = line.split()
                    intensities_per_file.append([float(val) for val in data[1:]])

                # 添加每列数据整理后的样本特征
                X_data.append(np.array(intensities_per_file).T)  # 转置数据，每列作为一个样本的特征
                y_data.extend([substance]*len(intensities_per_file))  # 添加对应的标签

# 转换数据为NumPy数组
X_data = np.array(X_data)
y_data = np.array(y_data)

# 划分数据集为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.2, random_state=42)

# 创建CNN模型
model = tf.keras.Sequential([
    tf.keras.layers.Conv1D(32, kernel_size=3, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2])),
    tf.keras.layers.MaxPooling1D(pool_size=2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(3, activation='softmax')
])

# 编译模型
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 训练模型
history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))
# 获取训练过程中的损失和准确度
train_loss = history.history['loss']
val_loss = history.history['val_loss']
train_acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
epochs = range(1, len(train_loss) + 1)

# 绘制损失值
plt.plot(epochs, train_loss, 'b', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# 绘制准确度
plt.plot(epochs, train_acc, 'b', label='Training accuracy')
plt.plot(epochs, val_acc, 'r', label='Validation accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()