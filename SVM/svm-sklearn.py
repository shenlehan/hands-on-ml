from sklearn.svm import SVC
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from tqdm import tqdm

def load_data():
  data = np.loadtxt('reference/第11章 支持向量机/spiral.csv', delimiter=',')
  print('dataset size: ', data.shape)
  x = data[:, :2]
  y = data[:, 2]

  return data, x, y

data, x, y = load_data()
model = SVC(kernel='rbf', gamma=50, tol=1e-6)
model.fit(x, y)
fig = plt.figure(figsize=(6,6))
G = np.linspace(-1.5, 1.5, 100)
G = np.meshgrid(G, G)
X = np.array([G[0].flatten(), G[1].flatten()]).T # 转换为每行一个向量的形式
Y = model.predict(X)
Y = Y.reshape(G[0].shape)

cmap = ListedColormap(['coral', 'royalblue'])
plt.contourf(G[0], G[1], Y, cmap=cmap, alpha=0.5)
plt.scatter(x[y == -1, 0], x[y == -1, 1], color='red', label='y=-1')
plt.scatter(x[y == 1, 0], x[y == 1, 1], marker='x', color='blue', label='y=1')
plt.xlabel(r'$x_1$')
plt.ylabel(r'$x_2$')
plt.legend()
plt.show()