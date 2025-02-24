import numpy as np
import matplotlib.pyplot as plt

# 设置随机数种子，以便结果可重现
np.random.seed(2)

# 模拟的 epoch 数
num_epochs = 100

# 初始化损失列表
losses = []

# 初始损失值
current_loss = 100.0

# 生成不规则的损失值
for epoch in range(num_epochs):
    # 在当前损失值上加入随机噪声
    noise = np.random.uniform(-5, 5)  # 随机噪声在-5到5之间
    current_loss += noise

    # 添加当前损失值到列表
    losses.append(current_loss)

# 绘制损失曲线图
plt.plot(range(num_epochs), losses, label="Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Irregular Loss Curve")
plt.legend()
plt.show()