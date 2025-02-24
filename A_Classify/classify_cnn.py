import numpy as np

# 读取原始数据
data = np.genfromtxt('./dataset/0ce6_0.txt', delimiter=' ')
print(data)
# 以 Numpy 数组的形式读取训练和测试数据
train_data = np.genfromtxt('./dataset/0ce6_0.txt', delimiter=',')
train_labels = np.genfromtxt('./dataset/0ce6_2-5.txt', delimiter=',')
test_data = np.genfromtxt('./dataset/0ce6_0.txt', delimiter=',')
test_labels = np.genfromtxt('./dataset/0ce6_2-5.txt', delimiter=',')

# 提取光谱数据
spectra = data[:, 0:3]
print(spectra)
# 将光谱数据reshape为(2068, 1)的形状
reshaped_data = np.reshape(spectra, (903, 1))
from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D, GlobalAveragePooling1D, Dense

# 构建模型
model = Sequential()

# 添加卷积层和池化层（共3次）
model.add(Conv1D(filters=16, kernel_size=4, activation='relu', input_shape=(2068, 1)))
model.add(Conv1D(filters=16, kernel_size=4, activation='relu'))
model.add(MaxPooling1D(pool_size=2))

model.add(Conv1D(filters=16, kernel_size=4, activation='relu'))
model.add(Conv1D(filters=16, kernel_size=4, activation='relu'))
model.add(MaxPooling1D(pool_size=2))

model.add(Conv1D(filters=16, kernel_size=4, activation='relu'))
model.add(Conv1D(filters=16, kernel_size=4, activation='relu'))
model.add(MaxPooling1D(pool_size=2))

# 添加全局平均池化层和输出层
model.add(GlobalAveragePooling1D())
model.add(Dense(3, activation='softmax'))

# 编译模型
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# 训练模型
model.fit(train_data, train_labels, epochs=10, batch_size=32, validation_data=(test_data, test_labels))