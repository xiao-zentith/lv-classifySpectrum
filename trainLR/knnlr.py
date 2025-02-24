import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

folder_path = "dataset_500-800/"
# folder_path = "dataset_normalization_maxmin/"
# folder_path = "dataset_normalization_mean/"
# # 类别名称和标签映射
categories = {
    # 'ce6_0': 1, 'ce6_2-5': 1, 'ce6_5': 1, 'ce6_10': 1, 'ce6_20': 1,
    'hpts_0': 0, 'hpts_2-5': 1, 'hpts_5': 2, 'hpts_10': 3, 'hpts_20': 4
    # 'Ru_0': 0, 'Ru_2-5': 0, 'Ru_5': 0, 'Ru_10': 0, 'Ru_20': 0
}
    # categories = {
#     'ce6_0': 2_0, 'ce6_2-5': 2_2.5, 'ce6_5': 2_5, 'ce6_10': 2_10, 'ce6_20': 2_20,
#     'hpts_0': 1_0, 'hpts_2-5': 1_2.5, 'hpts_5': 1_5, 'hpts_10': 1_10, 'hpts_20': 1_20
# }
# 读取数据并分配标签
all_data = []
for category, label in categories.items():
    data = np.loadtxt(f'{folder_path}{category}.txt')
    labeled_data = np.c_[data, label * np.ones(len(data))]
    all_data.append(labeled_data)

# 合并数据集
all_data = np.vstack(all_data)

# 分离特征和目标变量
X = all_data[:, 1:]  # 光谱波长作为特征
y = all_data[:, -1]  # 浓度标签

# 数据划分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)

# 训练模型
model = LinearRegression()
# model = SVR(kernel='rbf')
# model = RandomForestRegressor(n_estimators=100, random_state=42)
# # 创建神经网络模型
# model = Sequential([
#     Dense(64, activation='relu', input_shape=(20,)),
#     Dense(32, activation='relu'),
#     Dense(1)  # 输出层，因为是回归问题，所以输出一个值
# ])
# # 编译模型
# model.compile(optimizer='adam', loss='mean_squared_error')
# # 在模型训练过程中记录损失值
# history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test))
model.fit(X_train, y_train)

# 评估模型
accuracy = model.score(X_test, y_test)  # 这里使用模型的评分来简单评估准确性
print("准确度:", accuracy)

# 预测浓度
predicted_concentration = model.predict(X_test)
print("predicted_concentration", predicted_concentration)

# 计算均方误差
mse = mean_squared_error(y_test, predicted_concentration)
print(f"Mean Squared Error: {mse}")

# 计算决定系数
r_squared = r2_score(y_test, predicted_concentration)
print(f"R-squared: {r_squared}")

plt.scatter(y_test, predicted_concentration)
plt.xlabel("Actual Concentration")
plt.ylabel("Predicted Concentration")
plt.title("Actual vs Predicted Concentration")
plt.show()


# # 提取训练集和验证集上的损失值
# train_loss = history.history['loss']
# val_loss = history.history['val_loss']
#
# # 绘制损失曲线
# epochs = range(1, len(train_loss) + 1)
# plt.plot(epochs, train_loss, label='Training Loss')
# plt.plot(epochs, val_loss, label='Validation Loss')
# plt.title('Training and Validation Loss')
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.legend()
# plt.show()


