import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

PATH = 'reference/第12章 决策树/titanic'
data = pd.read_csv(os.path.join(PATH, 'train.csv'))
print(data.info())
print(data[:5])
data.drop(columns=['PassengerId', 'Name', 'Ticket'], inplace=True)

feat_ranges = {}
cont_feat = ['Age', 'Fare']
bins = 10

for feat in cont_feat:
  min_val = np.nanmin(data[feat])
  max_val = np.nanmax(data[feat])
  feat_ranges[feat] = np.linspace(min_val, max_val, bins).tolist()
  print(feat, ':')
  for spt in feat_ranges[feat]:
    print(f'{spt:.4f}')

cat_feat = ['Sex', 'Pclass', 'SibSp', 'Parch', 'Cabin', 'Embarked']
for feat in cat_feat:
  data[feat] = data[feat].astype('category') # 把离散类型转化成 category 类型，建立 离散值-数字 的映射
  print(f'{feat}: {data[feat].cat.categories}')
  data[feat] = data[feat].cat.codes.to_list()
  ranges = list(set(data[feat]))
  # or ranges = list(range(len(data[feat].cat.categories)))
  # but remember, NaN -> -1
  ranges.sort()
  feat_ranges[feat] = ranges

# 清洗，把空值设成-1
data.fillna(-1, inplace=True)
for feat in feat_ranges.keys():
  feat_ranges[feat] = [-1] + feat_ranges[feat]

np.random.seed(0)
feature_names = data.columns[1:]
label_name = data.columns[0]
ratio = 0.8
split = int(ratio * len(data))
data.reindex(np.random.permutation(data.index))
train_x = data[:split].drop(columns=['Survived']).to_numpy()
train_y = data['Survived'][:split].to_numpy()

test_x = data[split:].drop(columns=['Survived']).to_numpy()
test_y = data['Survived'][split:].to_numpy()

print('train size: ', len(train_x))
print('test size: ', len(test_x))
print('features cnt: ', train_x.shape[1])

class Node:
  def __init__(self):
    self.feat = None
    self.split = None
    self.child = []

class DecisionTree:
  def __init__(self, X, Y, feat_ranges, lbd):
    self.root = Node()
    self.X = X
    self.Y = Y
    self.feat_ranges = feat_ranges
    self.lbd = lbd
    self.eps = 1e-8
    self.T = 0
    self.ID3(self.rooot, self.X, self.Y)

  def aloga(self, a):
    return a * np.log2(a + self.eps)
  
  def entropy(self, Y):
    cnt = np.unique(Y, return_counts=True)[1]
    N = len(Y)
    ent = -np.sum([self.aloga(Ni / N) for Ni in cnt])
    return ent
  
  def info_gain(self, X, Y, feat, val):
    N = len(Y)
    if N == 0:
      return
    HX = self.entropy(Y)
    HXY = 0
    # compute H(X|X_F <= val) H(X|X_F > val)
    Y_l = Y[X[:, feat] <= val]
    HXY += len(Y_l) / len(Y) * self.entropy(Y_l)

    Y_r = Y[X[:, feat] > val]
    HXY += len(Y_r) / len(Y) * self.entropy(Y_r)

    return HX - HXY

  
  
  

  



if __name__ == '__main__':
  pass