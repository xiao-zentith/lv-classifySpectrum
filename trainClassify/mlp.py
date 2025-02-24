import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from tensorflow.python.keras.models import load_model
from tensorflow.python.keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping
import functionUtils

# folder_path = "dataset_500-800/"
# folder_path = "dataset_normalization_maxmin/"
folder_path = "dataset_normalization_mean/"
# # 类别名称和标签映射
categories = {
    'ce6_0': 1, 'ce6_2-5': 1, 'ce6_5': 1, 'ce6_10': 1, 'ce6_20': 1,
    'hpts_0': 2, 'hpts_2-5': 2, 'hpts_5': 2, 'hpts_10': 2, 'hpts_20': 2,
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
print(all_data.shape)

# 分离特征和目标变量
X = all_data[:, 1:-1]  # 光谱波长作为特征
y = all_data[:, -1]  # 类别标签

# 对类别标签进行编码
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

# 数据划分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
# 初始化PCA并拟合训练集
X_train, X_test = functionUtils.pca(2, X_train, X_test)
# 构建多层感知机模型
model = Sequential()
model.add(Dense(64, activation='relu', input_shape=(X_train.shape[1], )))
model.add(Dense(32, activation='relu'))
model.add(Dense(len(encoder.classes_), activation='softmax'))  # 输出层

# 编译模型
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# 设置模型训练时的回调函数
checkpoint = ModelCheckpoint('mlp_model.h5', monitor='val_loss', save_best_only=True)
early_stop = EarlyStopping(monitor='val_loss', patience=5)

# 训练模型
history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test), callbacks=[checkpoint, early_stop])

# 绘制损失曲线
train_loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(train_loss) + 1)

plt.plot(epochs, train_loss, label='Training Loss')
plt.plot(epochs, val_loss, label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# 评估模型
model = load_model('mlp_model.h5')  # 加载最佳模型
score = model.evaluate(X_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

