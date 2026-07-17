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

def simple_poly_kernel(d):
  def k(x, y):
    return np.inner(x, y) ** d
  return k

def rbf_kernel(sigma):
  def k(x, y):
    return np.exp(-np.inner(x - y, x - y) / (2.0 * sigma ** 2))
  return k

def cos_kernel(x, y):
  return np.inner(x, y) / (np.linalg.norm(x, 2) * np.linalg.norm(y, 2))

def sigmoid_kernel(beta, c):
  def k(x, y):
    return np.tanh(beta * np.inner(x, y) + c)
  return k

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

C = 1e8
max_iter = 1000
kernels = [
  simple_poly_kernel(3), 
  rbf_kernel(0.1), 
  cos_kernel, 
  sigmoid_kernel(1, -1)
]
ker_names = ['Poly(3)', 'RBF(0.1)', 'Cos', 'Sigmoid(1,-1)']
np.random.seed(0)
cmap = ListedColormap(['coral', 'royalblue'])

if __name__ == '__main__':
  data, x, y = load_data()
  fig, axes = plt.subplots(2, 2, figsize=(10, 10))
  axes = axes.flatten()

  for i, ax in enumerate(axes):  
    # orginal data
    print('核函数：', ker_names[i])
    alpha = SMO(x, y, kernels[i], C=C, max_iter=max_iter)
    sup_idx = alpha > 1e-5 # 支持向量的系数不为零
    sup_x = x[sup_idx] # 支持向量
    sup_y = y[sup_idx]
    sup_alpha = alpha[sup_idx]

    def wx(x_new):
      s = 0
      for xi, yi, ai in zip(sup_x, sup_y, sup_alpha):
        s += yi * ai * kernels[i](xi, x_new)
      return s
    
    neg = [wx(xi) for xi in sup_x[sup_y == -1]]
    pos = [wx(xi) for xi in sup_x[sup_y == 1]]
    b = -0.5 * (np.max(neg) + np.min(pos))

    G = np.linspace(-1.5, 1.5, 1000)
    G = np.meshgrid(G, G)
    X = np.array([G[0].flatten(), G[1].flatten()]).T # 转换为每行一个向量的形式
    Y = np.array([wx(xi) + b for xi in X])
    Y[Y < 0] = -1
    Y[Y >= 0] = 1
    Y = Y.reshape(G[0].shape)
    
    axes[i].scatter(x=x[y==-1, 0], y=x[y==-1, 1], s=30, color='red', marker='*', label='y=-1')
    axes[i].scatter(x=x[y==1, 0], y=x[y==1, 1], s=30, color='blue', marker='x', label='y=1')
    axes[i].contourf(G[0], G[1], Y, cmap=cmap, alpha=0.5)
    axes[i].set_title(ker_names[i])
    axes[i].set_xlabel(r'$x_1$')
    axes[i].set_ylabel(r'$x_2$')
    axes[i].legend()
  
  plt.show()
  plt.savefig('SVM/data.png')