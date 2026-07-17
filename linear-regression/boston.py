import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

def read_and_process_data():
  lines = np.loadtxt('reference/第4章 线性回归/USA_Housing.csv', delimiter=',', dtype=str)
  header = lines[0]
  lines = lines[1:].astype(float)
  print('数据特征：', '|'.join(header[:-1]))
  print('数据标签：', header[-1])
  print('数据总条数：', len(lines))

  ratio = 0.8
  split = int(ratio * len(lines))
  np.random.seed(0)
  lines = np.random.permutation(lines)
  train, test = lines[:split], lines[split:]

  scaler = StandardScaler()
  scaler.fit(train)

  train = scaler.transform(train)
  test = scaler.transform(test)

  x_train, y_train = train[:, :-1], train[:, -1]
  x_test, y_test = test[:, :-1], test[:, -1]

  return x_train, y_train, x_test, y_test

def manual_linreg(x_train, y_train, x_test, y_test):
  X = np.concat([x_train, np.ones((len(x_train), 1))], axis=-1)
  theta = np.linalg.inv(X.T @ X) @ X.T @ y_train
  X_test = np.concatenate([x_test, np.ones((len(x_test), 1))], axis=-1)
  y_pred = X_test @ theta
  rmse_loss = np.sqrt(np.square(y_test - y_pred).mean())
  print('Regression coeff: ', theta)
  print('RMSE: ', rmse_loss)

def sklearn_linreg(x_train, y_train, x_test, y_test):
  linreg = LinearRegression()
  linreg.fit(X=x_train, y=y_train)
  print('Regression coeff: ', linreg.coef_, linreg.intercept_)
  y_pred = linreg.predict(x_test)
  rmse_loss = np.sqrt(np.square(y_test - y_pred).mean())
  print('RMSE: ', rmse_loss)

def batch_generator(x, y, batch_size, shuffle=True):
  batch_count = 0
  if shuffle:
    idx = np.random.permutation(len(x))
    x = x[idx]
    y = y[idx]
    
  while True:
    start = batch_count * batch_size
    end = min((batch_count + 1) * batch_size, len(x))
    if start >= end:
      break

    batch_count += 1
    yield x[start: end], y[start: end]

def SGD(num_epoch, learning_rate, batch_size):
  X = np.concatenate([x_train, np.ones((len(x_train), 1))], axis=-1)
  X_test = np.concatenate([x_test, np.ones((len(x_test), 1))], axis=-1)
  theta = np.random.normal(size=X.shape[1])

  train_losses = []
  test_losses = []

  for i in range(num_epoch):
    batch_g = batch_generator(X, y_train, batch_size, shuffle=True)
    train_loss = 0
    for x, y in batch_g:
      grad = x.T @ (x @ theta - y)
      theta = theta - learning_rate * grad / len(x)
      train_loss += np.square(x @ theta - y).sum()

    train_loss = np.sqrt(train_loss / len(X))
    train_losses.append(train_loss)
    test_loss = np.sqrt(np.square(X_test @ theta - y_test).mean())
    test_losses.append(test_loss)

  print('Regression coeff: ', theta)
  return theta, train_losses, test_losses



if __name__ == '__main__':
  x_train, y_train, x_test, y_test = read_and_process_data()

  # manual_linreg(x_train, y_train, x_test, y_test)
  # sklearn_linreg(x_train, y_train, x_test, y_test)

  