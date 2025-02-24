# 读取txt文件，可以使用Pandas库的read_csv函数来读取。假设你的文件名叫做data.txt，代码如下：
import pandas as pd

data = pd.read_csv("./dataset/concat0.txt", sep="\t")  # txt文件用tab键分割
print(data)
# 将数据分为训练集和测试集。为了避免过拟合，我们需要将数据集分成训练集和测试集。可以使用train_test_split函数来进行划分。假设你想将数据集划分为80%的训练集和20%的测试集，代码如下：
from sklearn.model_selection import train_test_split

X = data.iloc[:, 2].values  # 提取光谱数据
y = data.iloc[:, 0].values  # 提取物质种类

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
# 使用kNN算法进行训练和预测。可以使用KNeighborsClassifier类来进行kNN分类器的训练和预测。假设你想使用k=5的算法，代码如下：
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)  # 训练模型
y_pred = knn.predict(X_test)  # 预测测试集
# 评估模型性能。可以使用混淆矩阵和分类报告来评估模型性能。代码如下：
from sklearn.metrics import confusion_matrix, classification_report

cm = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred)
print("混淆矩阵：\n", cm)
print("分类报告：\n", report)