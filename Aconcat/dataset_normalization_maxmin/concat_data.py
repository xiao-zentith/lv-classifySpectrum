import numpy as np

# 读取三个文件的光谱数据
file_paths = ["Ru_0.txt", "Ru_2-5.txt", "Ru_5.txt", "Ru_10.txt", "Ru_20.txt"]

# 存储光谱强度数据的列表
spectra_data = []

# 逐个读取文件，提取光谱强度数据
for file_path in file_paths:
    data = np.loadtxt(file_path)
    spectra = data[:, 1:]  # 提取光谱强度数据
    spectra_data.append(spectra)

# 在列方向上合并光谱强度数据
merged_spectra = np.hstack(spectra_data)

# 保存合并后的数据到新文件
output_data = np.column_stack((data[:, 0], merged_spectra))  # 合并光谱波长和强度数据
np.savetxt('Ru_merged_maxmin.txt', output_data, fmt='%1.4f', delimiter='\t')  # 保存为新文件
