import numpy as np
import matplotlib.pyplot as plt
import os

PATH = 'reference/第15章 主成分分析'

dataset = np.loadtxt(os.path.join(PATH, 'PCA_dataset.csv'), delimiter=',')
print('dataset size: ', len(dataset))

fig = plt.figure()
plt.scatter(x=dataset[:, 0], y=dataset[:, 1], color='red', marker='o', s=10)
plt.ylim(-2, 8)
plt.grid()
plt.show()
plt.savefig('PCA/figure1.png')

def pca(X, k):
  d, m = X.shape
  assert d > k, "d should be greater than k."
  
  X = X - np.mean(X, axis=0, keepdims=True)
  cov = X.T @ X
  eig_values, eig_vectors = np.linalg.eig(cov)
  idx = np.argsort(-eig_values)[:k]
  W = eig_vectors[:, idx]
  X = X @ W
  return X, W

X, W = pca(dataset, 2)
print('W:\n', W)

plt.figure()
plt.scatter(X[:, 0], X[:, 1], color='blue', s=10)
plt.axis('square')
plt.ylim(-5, 5)
plt.grid()
plt.xlabel(r'$x_1$')
plt.ylabel(r'$x_2$')
plt.show()
plt.savefig('PCA/figure2.png')