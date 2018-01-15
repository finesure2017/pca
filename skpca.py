import numpy as np
from sklearn.decomposition import PCA
if __name__ == "__main__":
    X = np.loadtxt('./data/testPCA4.txt', dtype=np.float)
    pca = PCA(n_components=1)
    pca.fit(X)
    new_data = pca.transform(X)
    origin_data = pca.inverse_transform(new_data)
    print('sklean降维成{}维数据后，前后方差比为：{}'.format(new_data.shape[1], np.var(origin_data) / np.var(X)))
