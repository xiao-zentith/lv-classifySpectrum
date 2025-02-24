# import os
# import matplotlib.pyplot as plt
# import numpy as np
#
# # 文件夹路径
# folder_path = r'D:\classifySpectrum\classify'
# output_path = r'D:\classifySpectrum\classify'
#
# # 获取文件夹中所有txt文件
# file_list = [file for file in os.listdir(folder_path) if file.endswith('.txt')]
#
# for file_name in file_list:
#     with open(os.path.join(folder_path, file_name), 'r') as file:
#         # 读取光谱数据
#         lines = file.readlines()
#         wavelengths = []
#         intensities = []
#         for line in lines:
#             data = line.split()
#             wavelengths.append(float(data[0]))
#             intensities.append([float(data[1])])
#
#         # 转换为NumPy数组以便处理
#         wavelengths = np.array(wavelengths)
#         intensities = np.array(intensities)
#
#         # 绘制图表
#         plt.figure()
#         for intensity in intensities.T:
#             plt.plot(wavelengths, intensity)
#         plt.xlabel('Wavelength')
#         plt.ylabel('Intensity')
#         # plt.title('Spectrum_' + file_name)
#
#         # 保存图像文件
#         image_name = file_name.replace('.txt', '.png')
#         plt.savefig(os.path.join(output_path, image_name))
#
#         # 可选：显示图表
#         # plt.show()
import os
import matplotlib.pyplot as plt
import numpy as np

# 文件夹路径
folder_path = r'D:\classifySpectrum\classify'
output_path = r'D:\classifySpectrum\classify'

# 获取文件夹中所有txt文件
file_list = [file for file in os.listdir(folder_path) if file.endswith('.txt')]

# 初始化图表
fig, ax = plt.subplots()

for file_name in file_list:
    with open(os.path.join(folder_path, file_name), 'r') as file:
        # 读取光谱数据
        lines = file.readlines()
        wavelengths = []
        intensities = []
        for line in lines:
            data = line.split()
            wavelengths.append(float(data[0]))
            intensities.append([float(data[1])])

        # 转换为NumPy数组以便处理
        wavelengths = np.array(wavelengths)
        intensities = np.array(intensities)

        # 绘制图表
        for intensity in intensities.T:
            ax.plot(wavelengths, intensity, label=file_name)

# 设置图表标签
ax.set_xlabel('Wavelength')
ax.set_ylabel('Intensity')
ax.legend()  # 添加图例

# 保存图像文件
plt.savefig(os.path.join(output_path, 'combined_spectra.png'))

# 可选：显示图表
plt.show()
