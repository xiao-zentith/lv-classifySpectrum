import numpy as np

# 类别名称和标签映射
categories = {
    'ce6_0': 1, 'ce6_2-5': 1, 'ce6_5': 1, 'ce6_10': 1, 'ce6_20': 1,
    'hpts_0': 2, 'hpts_2-5': 2, 'hpts_5': 2, 'hpts_10': 2, 'hpts_20': 2,
    'Ru_0': 3, 'Ru_2-5': 3, 'Ru_5': 3, 'Ru_10': 3, 'Ru_20': 3
}

# 读取数据并分配标签
all_data = []
for category, label in categories.items():
    spectra_data = np.loadtxt(f'{category}.txt')
    # 仅保留光谱强度列，进行数据增强并保留一位小数
    intensities = spectra_data[:, 1:]  # 光谱强度

    # 数据增强函数：噪声添加（Noise Addition）
    def add_noise(intensities, noise_level):
        noisy_intensities = np.copy(intensities)
        noise = np.random.normal(0, noise_level, size=noisy_intensities.shape)  # 生成随机噪声
        noisy_intensities += noise
        return np.round(noisy_intensities, 1)  # 保留一位小数

    # 数据增强函数：拉伸和压缩（Stretching and Compression）
    def stretch_intensity(intensities, stretch_factor):
        stretched_intensities = np.copy(intensities)
        stretched_intensities *= stretch_factor  # 对光谱强度进行拉伸或压缩
        return np.round(stretched_intensities, 1)  # 保留一位小数

    # 示例：应用数据增强操作
    noisy_intensities = add_noise(intensities, noise_level=0.1)  # 添加噪声
    stretched_intensities = stretch_intensity(intensities, stretch_factor=0.9)  # 拉伸光谱强度

    # 将增强后的光谱强度追加到原始文件的最后一列后面
    # 追加noisy_intensities到最后一列后面
    new_spectra_data = np.column_stack((spectra_data, noisy_intensities))
    # 选择使用noisy_intensities或stretched_intensities
    # new_spectra_data = np.column_stack((spectra_data, stretched_intensities))

    # 将更新后的数据保存到原始文件
    np.savetxt(f'{category}.txt', new_spectra_data, fmt='%.2f', delimiter='\t')
