# 定义一个空列表，存储所有文件读取的数据
all_cols = []

# 先读取第一个文件，获取每一列的数据
with open('1.TXT', 'r') as file1:
    # 读取第一个文件中的每一列数据
    cols = [line.split() for line in file1]

    # 将第一列数据添加到all_cols列表中
    first_col = [col[0] for col in cols]
    all_cols.append(first_col)

    # 将第二列数据添加到all_cols列表中
    second_col = [col[1] for col in cols]
    all_cols.append(second_col)

# 再读取剩下的数据文件，只获取第二列的数据
for i in range(2, 21):
    filename = f"{i}.TXT"
    with open(filename, 'r') as file:
        # 只读取每行的第二列数据
        second_col = [line.split()[1] for line in file]
        # 添加到all_cols列表中
        all_cols.append(second_col)

# 将所有数据写入新文件
with open('merged_data_Ru_20.txt', 'w') as outfile:
    # 计算总行数
    total_rows = len(first_col)

    # 循环处理每一行数据
    for row in range(total_rows):
        # 循环处理每一列数据
        for col in range(len(all_cols)):
            # 将数据写入新文件中，使用制表符(\t)分隔
            outfile.write(f"{all_cols[col][row]}\t")
        # 加一个换行符
        outfile.write('\n')

















