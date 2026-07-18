from sklearn import tree
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
data = data.reindex(np.random.permutation(data.index))
train_x = data[:split].drop(columns=['Survived']).to_numpy()
train_y = data['Survived'][:split].to_numpy()

test_x = data[split:].drop(columns=['Survived']).to_numpy()
test_y = data['Survived'][split:].to_numpy()

print('train size: ', len(train_x))
print('test size: ', len(test_x))
print('features cnt: ', train_x.shape[1])

c45 = tree.DecisionTreeClassifier(criterion='entropy', max_depth=6)
c45.fit(train_x, train_y)
cart = tree.DecisionTreeClassifier(criterion='gini', max_depth=6)
cart.fit(train_x, train_y)

c45_train_pred = c45.predict(train_x)
c45_test_pred = c45.predict(test_x)
cart_train_pred = cart.predict(train_x)
cart_test_pred = cart.predict(test_x)
print(f'训练集准确率：C4.5：{np.mean(c45_train_pred == train_y)}，' \
    f'CART：{np.mean(cart_train_pred == train_y)}')
print(f'测试集准确率：C4.5：{np.mean(c45_test_pred == test_y)}，' \
    f'CART：{np.mean(cart_test_pred == test_y)}')

from six import StringIO
import pydotplus

dot_data = StringIO()
tree.export_graphviz( # 导出sklearn的决策树的可视化数据
    c45,
    out_file=dot_data,
    feature_names=feature_names,
    class_names=['non-survival', 'survival'],
    filled=True, 
    rounded=True,
    impurity=False
)
# 用pydotplus生成图像
graph = pydotplus.graph_from_dot_data(
    dot_data.getvalue().replace('\n', '')) 
graph.write_png(os.path.join('decision-tree', 'tree.png'))