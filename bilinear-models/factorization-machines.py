import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics 
from tqdm import tqdm

data = np.loadtxt('reference/第7章 双线性模型/fm_dataset.csv', delimiter=',')

np.random.seed(0)
ratio = 0.8
split = int(ratio * len(data))
x_train = data[:split, :-1]
y_train = data[:split, -1]
x_test = data[split:, :-1]
y_test = data[split:, -1]
feature_num = x_train.shape[1]
print('训练集大小：', len(x_train))
print('测试集大小：', len(x_test))
print('特征数：', feature_num)

class FM:
  def __init__(self, feature_num, vector_dim):
    self.theta0 = 0.0
    self.theta = np.zeros(feature_num)
    self.v = np.random.normal(size=(feature_num, vector_dim))
    self.eps = 1e-6

  def _logistic(self, x):
    return 1.0 / (1 + np.exp(-x))
  
  def pred(self, x):
    linear_term = self.theta0 + x @ self.theta
    square_of_sum = np.square(x @ self.v)
    sum_of_square = np.square(x) @ np.square(self.v)

    y_pred = y_pred = self._logistic(linear_term \
            + 0.5 * np.sum(square_of_sum - sum_of_square, axis=1))\
        
    y_pred = np.clip(y_pred, self.eps, 1 - self.eps)
    return y_pred
  
  def update(self, grad0, grad_theta, grad_v, lr):
    self.theta0 -= lr * grad0
    self.theta -= lr * grad_theta
    self.v -= lr * grad_v

vector_dim = 16
learning_rate = 0.01
lbd = 0.05
max_training_step = 200
batch_size = 32

np.random.seed(0)
model = FM(feature_num, vector_dim)

train_acc = []
test_acc = []
train_auc = []
test_auc = []
vector_dim = 16
learning_rate = 0.01
lbd = 0.05
max_training_step = 200
batch_size = 32

with tqdm(range(max_training_step)) as pbar:
  for epoch in pbar:
    st = 0
    while st < len(x_train):
      ed = min(st + batch_size, len(x_train))
      X = x_train[st: ed]
      Y = y_train[st: ed]
      st += batch_size
      y_pred = model.pred(X)
      cross_entropy = -Y * np.log(y_pred) \
          - (1 - Y) * np.log(1 - y_pred)
      loss = np.sum(cross_entropy)
      grad_y = (y_pred - Y).reshape(-1, 1)
      grad0 = np.sum(grad_y * (1 / len(X) + lbd))
      grad_theta = np.sum(grad_y * (X / len(X) \
          + lbd * model.theta), axis=0)
      grad_v = np.zeros((feature_num, vector_dim))
      for i, x in enumerate(X):
        xv = x @ model.v
        grad_vi = np.zeros((feature_num, vector_dim))
        for s in range(feature_num):
          grad_vi[s] += x[s] * xv - (x[s] ** 2) * model.v[s]
        grad_v += grad_y[i] * grad_vi
      grad_v = grad_v / len(X) + lbd * model.v
      model.update(grad0, grad_theta, grad_v, learning_rate)

      pbar.set_postfix({
        '训练轮数': epoch,
        '训练损失': f'{loss:.4f}',
        '训练集准确率': train_acc[-1] if train_acc else None,
        '测试集准确率': test_acc[-1] if test_acc else None
      })
    y_train_pred = (model.pred(x_train) >= 0.5)
    acc = np.mean(y_train_pred == y_train)
    train_acc.append(acc)
    auc = metrics.roc_auc_score(y_train, y_train_pred) # sklearn中的AUC函数
    train_auc.append(auc)

    y_test_pred = (model.pred(x_test) >= 0.5)
    acc = np.mean(y_test_pred == y_test)
    test_acc.append(acc)
    auc = metrics.roc_auc_score(y_test, y_test_pred)
    test_auc.append(auc)

print(f'测试集准确率：{test_acc[-1]}，\t测试集AUC：{test_auc[-1]}')

plt.figure(figsize=(13, 5))
x_plot = np.arange(len(train_acc)) + 1

plt.subplot(121)
plt.plot(x_plot, train_acc, color='blue', label='train acc')
plt.plot(x_plot, test_acc, color='red', ls='--', label='test acc')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(122)
plt.plot(x_plot, train_auc, color='blue', label='train AUC')
plt.plot(x_plot, test_auc, color='red', ls='--', label='test AUC')
plt.xlabel('Epoch')
plt.ylabel('AUC')
plt.legend()
plt.savefig('BilinearModels/fm-loss.png')
plt.show()