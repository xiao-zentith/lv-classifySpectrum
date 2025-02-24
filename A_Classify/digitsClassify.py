# from sklearn.datasets import load_digits
# from sklearn.decomposition import PCA
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns; sns.set()
# digits = load_digits()
# print(digits.data.shape)
# #将数据投影到2维（可以自主决定投影维度）
# pca = PCA(2)  # project from 64 to 2 dimensions
# projected = pca.fit_transform(digits.data)
# print(digits.data.shape)
# print(projected.shape)
# plt.scatter(projected[:, 0], projected[:, 1],
#             c=digits.target, edgecolor='none', alpha=0.5,
#             cmap=plt.cm.get_cmap('Accent', 10))
# plt.xlabel('component 1')
# plt.ylabel('component 2')
# plt.colorbar()
# plt.show()
## pca特征降维
# 导入相关模块
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from numpy.linalg import eig
from sklearn.datasets import load_iris
iris = load_iris() # 导入矩阵，行是样本，列是指标
#X = np.array([[5.1, 3.5, 1.4, 0.2],
#                [4.9, 3, 1.4, 0.2]])
#自己导入矩阵数据可以用上面的注释代码，然后把X = iris.data 删掉即可
X = iris.data
# Standardize by remove average通过去除平均值进行标准化
X = X - X.mean(axis=0)# Calculate covariance matrix:计算协方差矩阵：
X_cov = np.cov(X.T, ddof=0)# Calculate  eigenvalues and eigenvectors of covariance matrix
# 计算协方差矩阵的特征值和特征向量
eigenvalues, eigenvectors = eig(X_cov)
pi = eigenvalues/np.sum(eigenvalues) #计算贡献率
p = np.cumsum(pi) #计算累计贡献率k=np.min(np.argwhere(p > 0.95))+1 #返回达到累计贡献率的阈值的下标# top k large eigenvectors选取前k个特征向量
k=2
klarge_index = eigenvalues.argsort()[-k:][::-1]
k_eigenvectors = eigenvectors[klarge_index]# X和k个特征向量进行点乘
X_pca = np.dot(X, k_eigenvectors.T)
print(X_pca) #输出主成分结果