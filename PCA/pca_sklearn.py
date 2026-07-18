import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.decomposition import PCA

PATH = 'reference/第15章 主成分分析'

dataset = np.loadtxt(os.path.join(PATH, 'PCA_dataset.csv'), delimiter=',')
print('dataset size: ', len(dataset))

X = dataset - np.mean(dataset, axis=0, keepdims=True)
pca_res = PCA(n_components=2).fit(X)
W = pca_res.components_.T
print(f'W: {W}')
x_pca = X @ W

plt.figure()
plt.scatter(x_pca[:, 0], x_pca[:, 1], color='blue', s=10)
plt.axis('square')
plt.ylim(-5, 5)
plt.grid()
plt.xlabel(r'$x_1$')
plt.ylabel(r'$x_2$')
plt.show()
plt.savefig('PCA/figure3.png')