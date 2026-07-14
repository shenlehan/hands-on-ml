from matplotlib.colors import ListedColormap
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.neighbors import KNeighborsClassifier

def visualize(m_x, m_y):
  plt.figure()
  plt.scatter(
    x=m_x[m_y == 0, 0], 
    y=m_x[m_y == 0, 1],
    c='blue',
    marker='o'
  )
  plt.scatter(
    x=m_x[m_y == 1, 0], 
    y=m_x[m_y == 1, 1],
    c='red',
    marker='x'
  )
  plt.xlabel('X axis')
  plt.ylabel('Y axis')
  plt.show()
  plt.savefig('KNN/knn_2.png')

def process_data():
  data = np.loadtxt('reference/第3章 k近邻算法/gauss.csv', delimiter=',')
  m_x, m_y = data[:, :2], data[:, 2]
  print(data.shape, m_x.shape, m_y.shape, type(m_x))

  # split test set
  ratio = 0.8
  split = int(len(m_x) * ratio)
  idx = np.random.permutation(np.arange(len(m_x)))

  m_x = m_x[idx]
  m_y = m_y[idx]
  x_train, x_test = m_x[:split], m_x[split:]
  y_train, y_test = m_y[:split], m_y[split:]

  return m_x, m_y, x_train, x_test, y_train, y_test

def distance(x: np.array, y: np.array):
  return np.mean((x - y) ** 2)


if __name__ == '__main__':
  np.random.seed(0)
  m_x, m_y, x_train, x_test, y_train, y_test = process_data()

  visualize(m_x, m_y)
  
  # we split the figure into grids in order to plot
  step = 0.02
  x_min, x_max = np.min(x_train[:, 0], axis=0) - 1, np.max(x_train[:, 0], axis=0) + 1
  y_min, y_max = np.min(x_train[:, 1], axis=0) - 1, np.max(x_train[:, 1], axis=0) + 1
  xx, yy = np.meshgrid(np.arange(x_min, x_max, step), np.arange(y_min, y_max, step))
  grid_data = np.concatenate([xx.reshape(-1, 1), yy.reshape(-1, 1)], axis=1)


  fig = plt.figure(figsize=(16,4.5))
  ks = [1, 3, 10]
  # cmap_light = ListedColormap(['royalblue', 'lightcoral'])
  cmap_prob = plt.cm.RdBu_r
  
  for i, k in enumerate(ks):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(x_train, y_train)
    # z = knn.predict(grid_data)
    z = knn.predict_proba(grid_data)[:, 1] # get the probabliity of z=1

    ax = fig.add_subplot(1, 3, i + 1)
    # ax.pcolormesh(xx, yy, z.reshape(xx.shape), cmap=cmap_light, alpha=0.7)
    mesh = ax.pcolormesh(xx, yy, z.reshape(xx.shape), cmap=cmap_prob, alpha=0.7, vmin=0, vmax=1)
    ax.scatter(x_train[y_train == 0, 0], x_train[y_train == 0, 1], c='blue', marker='o')
    ax.scatter(x_train[y_train == 1, 0], x_train[y_train == 1, 1], c='red', marker='x')

    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_title(f'K = {k}')
  
  fig.colorbar(mesh, ax=fig.axes, orientation='horizontal', fraction=0.05, label='Probability of Class 1 (Red)')
  plt.show()
  plt.savefig('KNN/knn_4.png')