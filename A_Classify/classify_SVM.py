import numpy as np

# 初始化一个二维数组，用于存储每个物质20次测量的光谱强度
data = []
txt_files=['D:\pythonProject\dataset\ce6_0.txt','D:\pythonProject\dataset\hpts_0.txt','D:\pythonProject\dataset\Ru_0.txt']
# 循环读取每个txt文件
for file_name in txt_files:
    # 读取txt文件中的数据
    measurements = np.loadtxt(file_name, delimiter='\t', skiprows=1)

    # 计算20次测量的平均值
    average_measurement = np.mean(measurements[:, 1:], axis=1)

    # 将平均值添加到data数组中
    data.append(average_measurement)

# 将data转换为二维数组
data = np.array(data)

# 输出整理后的数据格式
print(data)




# from sklearn.svm import SVC
# from sklearn.model_selection import train_test_split
#
# X = [[0.1, 0.2, 0.15, 0.3, 0.35, 0.2, 0.15],  # 物质1的测试结果
#      [0.3, 0.25, 0.35, 0.15, 0.3, 0.35, 0.2],  # 物质2的测试结果
#      [0.4, 0.5, 0.45, 0.15, 0.3, 0.35, 0.2]]   # 物质3的测试结果
#
# Y = [0, 1, 2]   # 标签，表示每个样本所属的类别
# # 划分训练集和测试集
# X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
# print(X_train, X_test)
# print(y_train, y_test)

# # 创建SVM分类器
# clf = SVC()
#
# # 拟合模型
# clf.fit(X_train, y_train)
#
# # 预测
# y_pred = clf.predict(X_test)
#
# # 打印预测结果
# print(y_pred)