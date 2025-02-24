import numpy as np

# 假设data包含光谱数据，第一列为光谱波长，其余列为光谱强度
categories = {
    'ce6_0': 1, 'ce6_2-5': 1, 'ce6_5': 1, 'ce6_10': 1, 'ce6_20': 1,
    'hpts_0': 2, 'hpts_2-5': 2, 'hpts_5': 2, 'hpts_10': 2, 'hpts_20': 2,
    'Ru_0': 0, 'Ru_2-5': 0, 'Ru_5': 0, 'Ru_10': 0, 'Ru_20': 0
}

# 读取数据并分配标签
all_data = []
for category, label in categories.items():
    data = np.loadtxt(f'{category}.txt')

    # 提取光谱强度列
    wavelength = data[:, 0]
    spectra = data[:, 1:]

    # 对光谱强度进行标准化处理
    mean = np.mean(spectra, axis=0)
    std = np.std(spectra, axis=0)
    standardized_spectra = (spectra - mean) / std

    # 将归一化后的数据覆盖写入原始文件
    data[:, 1:] = standardized_spectra

    # 将数据写入文件
    np.savetxt(f'{category}.txt', np.column_stack((wavelength, standardized_spectra)), fmt=['%.1f'] + ['%.6f'] * (data.shape[1] - 1), delimiter='\t')  # 保存为txt文件，使用制表符作为分隔符
