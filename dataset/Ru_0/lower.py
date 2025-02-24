import os

# 文件夹路径
folder_path = 'D:\classifySpectrum\dataset\Ru_20'  # 文件夹路径

# 遍历文件夹中的所有文件
for filename in os.listdir(folder_path):
    if filename.endswith(".TXT"):
        old_file = os.path.join(folder_path, filename)  # 原始文件的完整路径
        new_file = os.path.join(folder_path, filename.lower())  # 新文件的完整路径

        # 重命名文件
        os.rename(old_file, new_file)

print("文件后缀名修改完成！")