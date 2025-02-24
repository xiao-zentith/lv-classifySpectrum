import os

# 原始文件路径
original_file_path = '/home/asus515/Downloads/classifySpectrum/dataset/F/50.TXT'
# 存放新文件的目录
output_dir = '/home/asus515/Downloads/classifySpectrum/dataset/F_20'

# 确保输出目录存在
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 初始化变量
current_file_index = 0
write = False

# 打开原始txt文件
with open(original_file_path, 'r', encoding='utf-8') as file:
    while True:
        line = file.readline()
        # print(line)
        if not line:  # 检查是否到达文件末尾
            print("所有数据段提取完成")
            break

        # 检查是否到达数据段的开始
        if 'nm	Data' in line:
            current_file_index += 1
            write = True
            # 创建新的txt文件
            new_file_path = os.path.join(output_dir, f'{current_file_index}.txt')
            new_file = open(new_file_path, 'w', encoding='utf-8')
            print(f'正在创建新文件：{new_file_path}')

        # 检查是否到达数据段的结束
        elif 'Sample:' in line and write:
            write = False
            new_file.close()  # 关闭当前数据文件
            print(f'已关闭文件：{new_file_path}')

        # 如果处于数据段中，则写入当前打开的新文件
        if write and not 'nm	Data' in line:
            new_file.write(line)

print("所有数据段提取完成，并分别保存在 'extracted_data_files' 目录下。")