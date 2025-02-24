import numpy as np

# 读取txt文件，并将数据存储为矩阵
data = np.loadtxt('2Ru_20.txt')

# 提取第三列及后面的所有列，并计算平均值
cols_to_average = data[:, 2:]
average_values = np.mean(cols_to_average, axis=1)

# 将平均值写入第三列位置
data[:, 2] = average_values

# 删除其他列，只保留第一列和第二列
data = data[:, :3]

# 将结果写回txt文件
fmt = ['%.0f', '%.0f', '%.2f']
np.savetxt('2Ru_20.txt', data, fmt=fmt, delimiter=' ')

# import numpy as np
# import os
#
# # 设置文件夹路径
# folder_path = '../cnn'
#
# # 获取文件夹中的所有txt文件
# txt_files = [f for f in os.listdir(folder_path) if f.endswith('.txt')]
#
# # 创建一个空列表来存储每个平均值
# average_values = []
#
# for file in txt_files:
#     file_path = os.path.join(folder_path, file)
#
#     # 读取txt文件，并将数据存储为矩阵
#     data = np.loadtxt(file_path)
#
#     # 提取第三列及后面的所有列，并计算平均值
#     cols_to_average = data[:, 2:]
#     average_value = np.mean(cols_to_average)
#
#     # 将每个平均值添加到列表中
#     average_values.append(average_value)
#
#     # 将原文件内容修改为只保留第一第二列以及平均值
#     new_data = np.column_stack((data[:, :2], average_value))
#     np.savetxt(file_path, new_data, fmt='%.3f', delimiter='\t')