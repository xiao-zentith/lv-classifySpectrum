# 打开原始文件
with open('Ru_20.txt', 'r', encoding='utf-8') as f:
    # 读取原始内容并拆分为行
    lines = f.readlines()

# 定义新列的值
new_column = [2 for _ in range(len(lines))] # 创建与原始行数相同长度的新列

# 在每一行的开头添加新列
new_lines = [str(new_column[i]) + '\t' + lines[i] for i in range(len(lines))]

# 写入新文件
with open('../../A_Classify/dataset/2Ru_20.txt', 'w', encoding='utf-8') as f:
    for line in new_lines:
        f.write(line)
