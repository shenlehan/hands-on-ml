import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

def read_and_process_data():
  data = np.loadtxt('reference/第6章 逻辑斯谛回归/lr_dataset.csv', delimiter=',')
  X, y = data[:, :-1], data[:, -1].astype(int)
  print(X.shape, y.shape, X.dtype, y.dtype)

  ratio = 0.7
  split = int(ratio * len(data))
  np.random.seed(0)
  data = np.random.permutation(data)
  train, test = data[:split], data[split:]

  fig = plt.figure(figsize=(6, 4))
  plt.scatter(x=X[y==0, 0], y=X[y==0, 1], marker='P', color='coral', s=10)
  plt.scatter(x=X[y==1, 0], y=X[y==1, 1], marker='x', color='blue', s=10)
  plt.xlabel('X1 axis')
  plt.ylabel('X2 axis')
  plt.savefig('LogisticRegression/lr_1.png')
  plt.show()

  x_train, y_train = train[:, :-1], train[:, -1]
  x_test, y_test = test[:, :-1], test[:, -1]

  return x_train, y_train, x_test, y_test

if __name__ == '__main__':
  x_train, y_train, x_test, y_test = read_and_process_data()
  lr_clf = LogisticRegression(solver='liblinear')
  lr_clf.fit(x_train, y_train)
  print('Regression coeff: ', lr_clf.coef_[0], lr_clf.intercept_)
  y_pred = lr_clf.predict(x_test)
  print('accuracy: ',np.mean(y_pred == y_test))