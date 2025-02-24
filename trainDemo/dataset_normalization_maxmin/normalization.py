import numpy as np

# # 假设data包含光谱数据，第一列为光谱波长，其余列为光谱强度
# categories = {
#     'ce6_0': 1, 'ce6_2-5': 1, 'ce6_5': 1, 'ce6_10': 1, 'ce6_20': 1,
#     'hpts_0': 2, 'hpts_2-5': 2, 'hpts_5': 2, 'hpts_10': 2, 'hpts_20': 2,
#     'Ru_0': 0, 'Ru_2-5': 0, 'Ru_5': 0, 'Ru_10': 0, 'Ru_20': 0
# }

categories = {
    'C6_0': 3, 'C6_2-5': 1, 'C6_5': 1, 'C6_10': 1, 'C6_20': 1,
    'F_0': 4, 'F_2-5': 4, 'F_5': 4, 'F_10': 4, 'F_20': 4,
}

# 读取数据并分配标签
all_data = []
for category, label in categories.items():
    data = np.loadtxt(f'{category}.txt')

    # 提取光谱强度列
    wavelength = data[:, 0]
    spectra = data[:, 1:]

    # 对光谱强度进行归一化处理
    normalized_spectra = (spectra - np.min(spectra, axis=0)) / (np.max(spectra, axis=0) - np.min(spectra, axis=0))

    # 将归一化后的数据覆盖写入原始文件
    data[:, 1:] = normalized_spectra

    # 将数据写入文件
    np.savetxt(f'{category}.txt', np.column_stack((wavelength, normalized_spectra)), fmt=['%.1f'] + ['%.6f'] * (data.shape[1] - 1), delimiter='\t')  # 保存为txt文件，使用制表符作为分隔符
