import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from tqdm import tqdm

def load_data():
  data = np.loadtxt('reference/第11章 支持向量机/linear.csv', delimiter=',')
  print('dataset size: ', data.shape)
  x = data[:, :2]
  y = data[:, 2]

  return data, x, y



# SMO算法：每次选取两个 alpha_i alpha_j，对二者做最优化（实质上就是二次函数问题）
# 反复迭代即可
def SMO(x, y, ker, C, max_iter):
  m = x.shape[0]
  alpha = np.zeros(m)
  K = np.zeros((m, m))

  # compute Gram matrix
  for i in range(m):
    for j in range(m):
      K[i, j] = ker(x[i], x[j])

  for l in range(max_iter):
    for i in range(m):
      j = np.random.choice([t for t in range(m) if t != i])

      eta = K[j, j] + K[i, i] - 2 * K[i, j]
      e_i = np.sum(y * alpha * K[:, i]) - y[i]
      e_j = np.sum(y * alpha * K[:, j]) - y[j]
      alpha_i = alpha[i] - y[i] * (e_i - e_j) / (eta + 1e-5)
      zeta = alpha[i] * y[i] + alpha[j] * y[j]

      if y[i] == y[j]:
        lower = max(0, zeta / y[i] - C)
        upper = min(C, zeta / y[i])
      else:
        lower = max(0, zeta / y[i])
        upper = min(C, zeta / y[i] + C)

      alpha_i = np.clip(alpha_i, lower, upper)
      alpha_j = (zeta - y[i] * alpha_i) / y[j]

      alpha[i], alpha[j] = alpha_i, alpha_j

  return alpha

def plot(w, b, sup_idx):
  fig = plt.figure(figsize=(10, 6))
  
  # orginal data
  plt.scatter(x=x[y==-1, 0], y=x[y==-1, 1], s=30, color='red', marker='*', label='y=-1')
  plt.scatter(x=x[y==1, 0], y=x[y==1, 1], s=30, color='blue', marker='x', label='y=1')

  # 注意最终解出来的 w b 是 w^Tx+b=0
  X = np.linspace(np.min(x[:, 0]), np.max(x[:, 0]), 100)
  Y = -(w[0] * X + b) / (w[1] + 1e-5)
  plt.plot(X, Y, color='black')

  plt.scatter(x[sup_idx, 0], x[sup_idx, 1], s=150, color='none', 
              marker='o', edgecolors='purple', label='support vectors')
  
  plt.xlabel(r'$x_1$')
  plt.ylabel(r'$x_2$')
  plt.legend()
  plt.show()
  plt.savefig('SVM/data.png')

C = 1e8
max_iter = 1000
np.random.seed(0)

if __name__ == '__main__':
  data, x, y = load_data()
  alpha = SMO(x, y, ker=np.inner, C=C, max_iter=max_iter)
  sup_idx = alpha > 1e-5
  print('支持向量个数：', np.sum(sup_idx))
  w = np.sum((alpha[sup_idx] * y[sup_idx]).reshape(-1, 1) * x[sup_idx], axis=0)
  wx = x @ w.reshape(-1, 1)
  b = -0.5 * (np.max(wx[y == -1]) + np.min(wx[y == 1]))
  print(f"w={w}, b={b}")

  plot(w, b, sup_idx)