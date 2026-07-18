import numpy as np
import matplotlib.pyplot as plt

dataset = np.loadtxt('reference/第14章 k均值聚类/kmeans_data.csv', delimiter=',')
print('dataset size: ', len(dataset))

def show_cluster(dataset, cluster, centroids=None):
  colors = ['blue', 'red', 'green', 'purple']
  markers = ['o', 'x', 's', 'd']

  K = len(np.unique(cluster))
  for i in range(K):
    plt.scatter(dataset[cluster==i, 0], dataset[cluster==i, 1], marker=markers[i], color=colors[i], s=10)
  
  if centroids is not None:
    plt.scatter(centroids[:, 0], centroids[:, 1], color=colors[:K], marker='+', s=150)

  plt.savefig('k-means/fig1.png')
  plt.show()

def random_init(dataset, K):
  idx = np.random.choice(np.arange(len(dataset)), size=K, replace=False)
  return dataset[idx]

def kmeans(dataset, K, init_cent):
  centroids = init_cent(dataset, K)
  cluster = np.zeros(len(dataset), dtype=int)
  changed = True

  itr = 0
  while changed:
    changed = False
    loss = 0
    for i, data in enumerate(dataset):
      dis = np.sum((centroids - data) ** 2, axis=-1)
      k = np.argmin(dis)
      if cluster[i] != k:
        cluster[i] = k
        changed = True
      
      loss += np.sum((data - centroids[k]) ** 2)

    print(f'Iteration {itr}, Loss {loss:.3f}')
    show_cluster(dataset, cluster, centroids)
    for i in range(K):
        centroids[i] = np.mean(dataset[cluster == i], axis=0)
    itr += 1

  return centroids, cluster

def kmeanspp_init(dataset, K):
  idx = np.random.choice(np.arange(len(dataset)))
  centroids = dataset[idx][None]
  # print(dataset.shape, centroids.shape,dataset[idx].shape)
  for k in range(1, K):
    d = []
    for data in dataset:
      dis = np.sum((centroids - data) ** 2, axis=-1)
      d.append(np.min(dis) ** 2)
    d = np.array(d)
    d /= np.sum(d)
    cent_id = np.random.choice(np.arange(len(dataset)), p=d)
    cent = dataset[cent_id]
    centroids = np.concatenate([centroids, cent[None]], axis=0)

  return centroids



if __name__ == '__main__':
  # show_cluster(dataset, np.zeros(len(dataset), dtype=int))
  np.random.seed(0)
  cent, cluster = kmeans(dataset, 4, kmeanspp_init)