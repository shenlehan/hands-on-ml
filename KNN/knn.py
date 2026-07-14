import numpy as np
import matplotlib.pyplot as plt
import os

def process_data():
  m_x = np.loadtxt('reference/第3章 k近邻算法/mnist_x', delimiter=' ')
  m_y = np.loadtxt('reference/第3章 k近邻算法/mnist_y')

  # print(m_x.shape, m_y.shape, type(m_x))

  data = m_x.reshape([-1, 28, 28])
  plt.figure()
  plt.imshow(data[0], cmap='gray')
  plt.savefig("KNN/knn_1.png")

  # split test set
  ratio = 0.8
  split = int(len(m_x) * ratio)
  idx = np.random.permutation(np.arange(len(m_x)))

  m_x = m_x[idx]
  m_y = m_y[idx]
  x_train, x_test = m_x[:split], m_x[split:]
  y_train, y_test = m_y[:split], m_y[split:]

  return x_train, x_test, y_train, y_test

def distance(x: np.array, y: np.array):
  return np.mean((x - y) ** 2)

class KNN:
  def __init__(self, k, label_num):
    self.k = k
    self.label_num = label_num

  def fit(self, x_train, y_train):
    self.x_train = x_train
    self.y_train = y_train

  def get_knn_indices(self, x):
    dis = list(map(lambda a: distance(a, x), self.x_train))
    knn_indices = np.argsort(dis)[:self.k]
    return knn_indices

  def get_label(self, x):
    knn_indices = self.get_knn_indices(x)
    cnt = np.zeros((self.label_num), dtype=int)
    for idx in knn_indices:
      label = int(self.y_train[idx])
      cnt[label] += 1

    return np.argmax(cnt)

  def predict(self, x_test):
    return [self.get_label(x) for x in x_test]


if __name__ == '__main__':
  np.random.seed(0)
  x_train, x_test, y_train, y_test = process_data()

  for k in range(1, 10):
    knn = KNN(k, label_num=10)
    knn.fit(x_train, y_train)
    predicted_labels = knn.predict(x_test)

    accuracy = np.mean(predicted_labels == y_test)
    print(f'K的取值为 {k}, 预测准确率为 {accuracy * 100:.1f}%')