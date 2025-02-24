
import numpy as np

# 加载包含光谱数据的txt文件
file_path = "dataset/ce6_0.txt"  # 替换为你的文件路径
spectra = np.loadtxt(file_path)

wavelengths = spectra[:, 0]  # 提取光谱波长列
intensities = spectra[:, 1:]  # 提取光谱强度列

mean_intensity = np.mean(intensities, axis=0)

k = np.sum(intensities * mean_intensity, axis=0) / np.sum(mean_intensity ** 2)
b = np.mean(intensities - k * mean_intensity, axis=0)

corrected_spectra = (intensities - b) / k

corrected_spectra_with_wavelength = np.insert(corrected_spectra, 0, wavelengths, axis=1)

# 保存校正后的光谱数据到新的txt文件
output_file_path = "corrected_spectra.txt"  # 新文件的路径和名称
np.savetxt(output_file_path, corrected_spectra_with_wavelength, fmt='%.2f')
# import numpy as np
# from sklearn.linear_model import LinearRegression
#
# # 假设spectra是原始数据的光谱矩阵，其中每一行代表一个样本的光谱
# # 第一列是光谱波长，后续列是对应的光谱强度
# file_path = "dataset/ce6_0.txt"  # 替换为你的文件路径
# spectra = np.loadtxt(file_path)
# wavelengths = spectra[:, 0]  # 提取光谱波长列
# # 计算所有样本光谱的平均光谱
# mean_spectrum = np.mean(spectra[:, 1:], axis=0)
#
# # 创建一个空数组用于存储每个样本的回归截距和斜率差
# intercepts_diff = []
# slopes_diff = []
#
# # 逐个样本计算一元线性回归的截距和斜率差
# for spectrum in spectra[:, 1:]:
#     # 使用sklearn进行线性回归拟合
#     model = LinearRegression()
#     model.fit(mean_spectrum.reshape(-1, 1), spectrum.reshape(-1, 1))
#     intercepts_diff.append(model.intercept_[0])
#     slopes_diff.append(model.coef_[0][0])
#
# # intercepts_diff和slopes_diff分别包含了每个样本的截距差和斜率差
# # 假设intercepts_diff和slopes_diff分别包含了每个样本的截距差和斜率差
#
# # 创建一个空数组用于存储校正后的数据
# corrected_spectra = []
#
# # 对每个样本的光谱进行校正
# for spectrum, intercept_diff, slope_diff in zip(spectra[:, 1:], intercepts_diff, slopes_diff):
#     corrected_spectrum = (spectrum - intercept_diff) / slope_diff
#     corrected_spectra.append(corrected_spectrum)
#
# corrected_spectra = np.array(corrected_spectra)
# # corrected_spectra 现在包含了校正后的数据
# corrected_spectra_with_wavelength = np.insert(corrected_spectra, 0, wavelengths, axis=1)
#
# # 保存校正后的光谱数据到新的txt文件
# output_file_path = "corrected_spectra.txt"  # 新文件的路径和名称
# np.savetxt(output_file_path, corrected_spectra_with_wavelength, fmt='%.2f')
