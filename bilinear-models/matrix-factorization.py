import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

data = np.loadtxt('reference/第7章 双线性模型/movielens_100k.csv', delimiter=',', dtype=int)
print('dataset size: ', len(data))
data[:, :2] = data[:, :2] - 1

users = set()
items = set()
for i, j, k in data:
  users.add(i)
  items.add(j)
user_num = len(users)
item_num = len(items)
print(f'用户数：{user_num}，电影数：{item_num}')

np.random.seed(0)

ratio = 0.8
split = int(ratio * len(data))
np.random.shuffle(data)
train, test = data[:split], data[split:]

print(train.shape, test.shape)

user_cnt = np.bincount(train[:, 0], minlength=user_num)
item_cnt = np.bincount(train[:, 1], minlength=user_num)

user_train, user_test = train[:, 0], test[:, 0]
item_train, item_test = train[:, 1], test[:, 1]
y_train, y_test = train[:, 2], test[:, 2]

assert(len(user_train) == len(train))

# 矩阵分解：设 A=Q^T R，直接梯度下降计算
class MF:
  def __init__(self, N, M, d):
    self.user_params = np.ones((N, d))
    self.item_params = np.ones((M, d))

  def pred(self, user_id, item_id):
    user_param = self.user_params[user_id]
    item_param = self.item_params[item_id]
    rating_pred = np.sum(user_param * item_param, axis=1)
    return rating_pred
  
  def update(self, user_grad, item_grad, lr):
    self.user_params -= lr * user_grad
    self.item_params -= lr * item_grad

def train(model, learning_rate, lbd, max_training_step, batch_size):
  train_losses = []
  test_losses = []

  batch_num = int(np.ceil(1.0 * len(user_train) / batch_size))
  pbar = tqdm(range(0, max_training_step))
  for epoch in pbar:
    train_rmse = 0
    for i in range(batch_num):
      st = i * batch_size
      ed = min(len(user_train), batch_size * (i + 1))
      user_batch = user_train[st: ed]
      item_batch = item_train[st: ed]
      y_batch = y_train[st: ed]

      y_pred = model.pred(user_batch, item_batch)
      P = model.user_params
      Q = model.item_params
      errs = y_batch - y_pred
      P_grad = np.zeros_like(P)
      Q_grad = np.zeros_like(Q)

      # 累加梯度
      for user, item, err in zip(user_batch, item_batch, errs):
        P_grad[user] = P_grad[user] - err * Q[item] + lbd * P[user]
        Q_grad[item] = Q_grad[item] - err * P[user] + lbd * Q[item]
      model.update(P_grad / len(user_batch), Q_grad / len(user_batch), learning_rate)
      train_rmse += np.mean(errs ** 2)

      pbar.set_postfix({
        'Epoch': epoch,
        'Train RMSE': f'{np.sqrt(train_rmse / (i + 1)):.4f}',
        'Test RMSE': f'{test_losses[-1]:.4f}' if test_losses else None
      })
  
    train_rmse = np.sqrt(train_rmse / len(user_train))
    train_losses.append(train_rmse)
    y_test_pred = model.pred(user_test, item_test)
    test_rmse = np.sqrt(np.mean((y_test - y_test_pred) ** 2))
    test_losses.append(test_rmse)

  return train_losses, test_losses



if __name__ == '__main__':
  feature_num = 16 # 特征数
  learning_rate = 0.1 # 学习率
  lbd = 1e-4 # 正则化强度
  max_training_step = 30
  batch_size = 64 # 批量大小

  # 建立模型
  model = MF(user_num, item_num, feature_num)
  # 训练部分
  train_losses, test_losses = train(model, learning_rate, lbd,
      max_training_step, batch_size)

  plt.figure()
  x = np.arange(max_training_step) + 1
  plt.plot(x, train_losses, color='blue', label='train loss')
  plt.plot(x, test_losses, color='red', ls='--', label='test loss')
  plt.xlabel('Epoch')
  plt.ylabel('RMSE')
  plt.legend()
  plt.savefig('BilinearModels/loss.png')
  plt.show()