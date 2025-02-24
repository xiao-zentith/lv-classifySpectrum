import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.signal import savgol_filter

# 画出单个光谱曲线
def plot_spectrum_demo(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()[1:]

    wavelengths = []
    intensities = []

    for line in lines:
        data = line.split()
        wavelengths.append(float(data[0]))
        intensities.append(float(data[1]))

    plt.figure(figsize=(8, 6))
    plt.plot(wavelengths, intensities, label='Spectrum')
    plt.xlabel('Wavelength')
    plt.ylabel('Intensity')
    plt.title('Spectrum Curve')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_all_spectra(file_path):
    data = np.loadtxt(file_path)

    wavelengths = data[:, 0]
    intensities = data[:, 1:]

    plt.figure(figsize=(8, 6))

    for i in range(intensities.shape[1]):
        plt.plot(wavelengths, intensities[:, i], label=f'Spectrum {i+1}')

    plt.xlabel('Wavelength')
    plt.ylabel('Intensity')
    plt.title('All Spectra Curves')
    plt.legend()
    plt.grid(True)
    plt.show()

# 对单个txt文件数据进行平滑，并保存成txt文件
def smooth_and_save_demo(file_path, window_size=11):
    # 读取数据
    data = np.loadtxt(file_path)

    # 提取波长和强度数据
    wavelengths = data[:, 0]
    intensities = data[:, 1:]

    # 对每列数据进行平滑处理
    smoothed_intensities = []
    for intensity_column in intensities.T:
        smoothed = savgol_filter(intensity_column, window_length=window_size, polyorder=3)
        smoothed_intensities.append(smoothed)

    # 合并波长和平滑后的强度数据
    smoothed_data = np.column_stack((wavelengths, np.array(smoothed_intensities).T))

    # 文件路径和名称
    new_file_path = 'smoothed_' + file_path.split('/')[-1]

    # 保存平滑后的数据
    np.savetxt(new_file_path, smoothed_data, fmt='%.6f', delimiter='\t', header='Wavelength\t' + '\t'.join([f'Smoothed_{i}' for i in range(1, data.shape[1])]), comments='')

    return new_file_path  # 返回新文件的路径

# 比较不同的窗口大小对数据的平滑结果，并计算均方误差
def plot_smoothed_spectra(file_path, window_sizes=[5, 7, 9, 11, 13], polyorder=3):
    # 读取数据
    with open(file_path, 'r') as file:
        lines = file.readlines()

    wavelengths = []
    intensities = []

    # 解析光谱数据
    for line in lines:
        data = line.split()
        wavelengths.append(float(data[0]))
        intensities.append(float(data[1]))

    # 转换为 NumPy 数组
    wavelengths = np.array(wavelengths)
    intensities = np.array(intensities)

    # 计算子图布局
    num_plots = len(window_sizes)
    num_cols = 2  # 可以根据需要调整子图的列数
    num_rows = 3  # 根据窗口大小数决定子图行数

    # 创建子图
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(12, 8))
    axes = axes.flatten()

    # 绘制原始光谱
    axes[0].plot(wavelengths, intensities, label='Original Spectrum')
    axes[0].set_title('Original Spectrum')
    axes[0].set_xlabel('Wavelength')
    axes[0].set_ylabel('Intensity')
    axes[0].legend()

    # 使用不同的窗口大小进行平滑处理，并绘制平滑后的曲线
    for i, window_size in enumerate(window_sizes):
        smoothed_intensities = savgol_filter(intensities, window_length=window_size, polyorder=polyorder)

        # 计算均方根误差
        rmse = np.sqrt(np.mean((smoothed_intensities - intensities) ** 2))
        print(f"Window Size {window_size}: RMSE = {rmse}")

        axes[i + 1].plot(wavelengths, smoothed_intensities, label=f'Window Size {window_size}')
        axes[i + 1].set_title(f'Smoothed (Window Size {window_size})')
        axes[i + 1].set_xlabel('Wavelength')
        axes[i + 1].set_ylabel('Intensity')
        axes[i + 1].legend()

    plt.tight_layout()
    plt.show()

    # 合并波长和平滑后的强度数据
    smoothed_data = np.column_stack((wavelengths, smoothed_intensities))

    # 文件路径和名称
    save_file_path = 'smoothed_data.txt'

    # 保存数据
    np.savetxt(save_file_path, smoothed_data, fmt='%.6f', delimiter='\t', header='Wavelength\tIntensity', comments='')

    return save_file_path

# 对所有的txt文件进行平滑处理并保存
def smooth_and_save_folder(folder_path, window_size=11):
    # 创建用于保存平滑文件的新文件夹
    smooth_folder = os.path.join(folder_path, 'smooth_'+str(window_size))
    os.makedirs(smooth_folder, exist_ok=True)  # 如果文件夹已存在则不重新创建

    # 获取文件夹内所有文件
    file_list = os.listdir(folder_path)
    txt_files = [file for file in file_list if file.endswith('.txt')]

    for txt_file in txt_files:
        file_path = os.path.join(folder_path, txt_file)
        data = np.loadtxt(file_path)

        wavelengths = data[:, 0]
        intensities = data[:, 1:]

        smoothed_intensities = []
        for intensity_column in intensities.T:
            smoothed = savgol_filter(intensity_column, window_length=window_size, polyorder=3)
            smoothed_intensities.append(smoothed)

        smoothed_data = np.column_stack((wavelengths, np.array(smoothed_intensities).T))

        # 保存到新的smooth文件夹中
        new_file_path = os.path.join(smooth_folder, f'smoothed_{txt_file}')
        np.savetxt(new_file_path, smoothed_data, fmt='%.6f', delimiter='\t', header='Wavelength\t' + '\t'.join([f'Smoothed_{i}' for i in range(1, data.shape[1])]), comments='')

    print("All files smoothed and saved in the 'smooth' folder.")

# 将数据进行最大最小值归一化并保存
def normalize_and_maxmin(data_folder, output_folder):
    # Create a new folder to store the normalized files
    os.makedirs(output_folder, exist_ok=True)

    categories = {
        'ce6_0': 1, 'ce6_2-5': 1, 'ce6_5': 1, 'ce6_10': 1, 'ce6_20': 1,
        'hpts_0': 2, 'hpts_2-5': 2, 'hpts_5': 2, 'hpts_10': 2, 'hpts_20': 2,
        'Ru_0': 0, 'Ru_2-5': 0, 'Ru_5': 0, 'Ru_10': 0, 'Ru_20': 0
    }
    new_categories = {f'smoothed_{key}': value for key, value in categories.items()}

    for category, label in new_categories.items():
        # Read data
        data = np.loadtxt(os.path.join(data_folder, f'{category}.txt'), skiprows=1)

        # Extract wavelength and spectra
        wavelength = data[:, 0]
        spectra = data[:, 1:]

        # Normalize spectra
        normalized_spectra = (spectra - np.min(spectra, axis=0)) / (np.max(spectra, axis=0) - np.min(spectra, axis=0))

        # Override the original file with normalized data
        data[:, 1:] = normalized_spectra

        # Save normalized data to a new file in the output folder
        output_file = os.path.join(output_folder, f'{category}_normalized.txt')
        np.savetxt(output_file, np.column_stack((wavelength, normalized_spectra)),
                   fmt=['%.1f'] + ['%.6f'] * (data.shape[1] - 1), header='', delimiter='\t')

# 均值归一化
def normalize_and_mean(data_folder_path, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    categories = {
        'ce6_0': 1, 'ce6_2-5': 1, 'ce6_5': 1, 'ce6_10': 1, 'ce6_20': 1,
        'hpts_0': 2, 'hpts_2-5': 2, 'hpts_5': 2, 'hpts_10': 2, 'hpts_20': 2,
        'Ru_0': 0, 'Ru_2-5': 0, 'Ru_5': 0, 'Ru_10': 0, 'Ru_20': 0
    }
    new_categories = {f'smoothed_{key}': value for key, value in categories.items()}
    for category, label in new_categories.items():
        data = np.loadtxt(os.path.join(data_folder_path, f'{category}.txt'), skiprows=1)

        # 提取光谱强度列
        wavelength = data[:, 0]
        spectra = data[:, 1:]

        # 对光谱强度进行标准化处理
        mean = np.mean(spectra, axis=0)
        std = np.std(spectra, axis=0)
        standardized_spectra = (spectra - mean) / std

        # 将归一化后的数据覆盖写入原始文件
        data[:, 1:] = standardized_spectra

        # Save normalized data to a new file in the output folder
        output_file = os.path.join(output_folder, f'{category}_normalized.txt')
        np.savetxt(output_file, np.column_stack((wavelength, standardized_spectra)),
                   fmt=['%.1f'] + ['%.6f'] * (data.shape[1] - 1), header='', delimiter='\t')


# data_folder_path = './dataset_500-800/smooth_11'  # Replace with the path to your data folder
# output_folder_path = './dataset_500-800/normalization_11_mean'  # Replace with the desired path for the output folder

# normalize_and_maxmin(data_folder_path, output_folder_path)
# normalize_and_mean(data_folder_path, output_folder_path)

# # 指定文件夹路径并调用函数进行处理
# folder_path = './dataset_500-800'  # 修改为您的文件夹路径
# smooth_and_save_folder(folder_path)

# 使用函数单个绘图
plot_spectrum_demo('./dataset_500-800/normalization_11/smoothed_ce6_0_normalized.txt')
# plot_all_spectra('D:\classifySpectrum\Aconcat\dataset_normalization_maxmin\Ru_merged_maxmin.txt')


# # 使用函数进行数据平滑
# smoothed_file_path = smooth_and_save_demo('./dataset_origin/ce6_0.txt')
# print(f"Smoothed data saved to {smoothed_file_path}")
#
# # 使用函数进行绘图和保存数据
# file_path = './dataset_origin/ce6_0.txt'
# smoothed_file_path = plot_smoothed_spectra(file_path)
# print(f"Smoothed data saved to {smoothed_file_path}")


# file_path = './dataset_origin/Ru_0.txt'
# import matplotlib.pyplot as plt
# import numpy as np
# from sklearn.decomposition import PCA
#
# # 读取txt文件数据到NumPy数组，假设文件名为 file_path
# data = np.loadtxt(file_path)
# # 转置数据，使每列代表一个样本
# spectra = data[:, 1:].T
# print(spectra.shape)
#
# # 创建并拟合PCA模型
# pca = PCA(n_components=15)  # 设置主成分数为20
# pca.fit(spectra)  # 拟合数据
#
# # 获取每个主成分的贡献率
# explained_var_ratio = pca.explained_variance_ratio_
# # # 输出降维后的数据形状
# reduced_data = pca.transform(spectra)
# print("降维后的数据形状:", reduced_data.shape)
# print(reduced_data)
# print(pca.components_)
# # 获取降维后的贡献值（方差解释度）
# explained_variance = pca.explained_variance_
# print("特征值：", explained_variance)
#
# # # 画出贡献值的累积贡献率
# explained_variance_ratio = pca.explained_variance_ratio_
# print("每个主成分方差的比例：", explained_variance_ratio)
# cumulative_variance_ratio = np.cumsum(explained_variance_ratio)
# print("累计比例：", cumulative_variance_ratio)
# # 绘制柱状图展示每个主成分的贡献率
# plt.figure(figsize=(8, 6))
# plt.bar(range(1, len(explained_variance_ratio) + 1), explained_variance_ratio)
# plt.xlabel('Principal Components')
# plt.ylabel('Variance Ratio')
# plt.title('Variance Ratio of Principal Components')
# plt.grid(True)
# plt.show()
#
# plt.figure(figsize=(8, 6))
# plt.plot(range(1, len(explained_variance) + 1), cumulative_variance_ratio, marker='o', linestyle='-')
# plt.xlabel('Number of Components')
# plt.ylabel('Cumulative Variance Ratio')
# plt.title('Cumulative Variance Ratio of Principal Components')
# plt.grid(True)
# plt.show()
#
# # 将降维后的数据转换回原始空间
# restored_data = pca.inverse_transform(reduced_data)
# # 打印原始光谱数据的维度
# print("原始光谱数据形状:", spectra.shape)
# print(restored_data.shape)
#
# # 提取并可视化每列光谱强度数据
# plt.figure(figsize=(14, 6))
# plt.subplot(1, 2, 1)
# plt.plot(data[:, 0], spectra.T)
# plt.xlabel('Wavelength')
# plt.ylabel('Intensity')
# plt.title('Original Spectrum')
#
# plt.subplot(1, 2, 2)
# plt.plot(data[:, 0], restored_data.T)
# plt.xlabel('Wavelength')
# plt.ylabel('Intensity')
# plt.title('Restored Spectrum')
# plt.tight_layout()
# plt.show()