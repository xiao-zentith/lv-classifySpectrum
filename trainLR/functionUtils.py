from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


def standard(X_train, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled


def pca(k, X_train, X_test):
    # 初始化PCA并拟合训练集
    pca = PCA(k)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)
    return X_train_pca, X_test_pca

