import numpy as np
import matplotlib.pyplot as plt
from skimage import io
from skimage.color import rgb2lab, lab2rgb
from sklearn.neighbors import KNeighborsRegressor
import os

block_size = 1
path = 'reference/第3章 k近邻算法/style_transfer'

def read_style_image(file_name, size=block_size):
  img = io.imread(file_name)
  fig = plt.figure()
  plt.imshow(img)
  plt.xlabel('X axis')
  plt.ylabel('Y axis')
  plt.show()

  img = rgb2lab(img)
  w, h = img.shape[:2]

  X = []
  Y = []

  for x in range(size, w - size):
    for y in range(size, h - size):
      X.append(
        img[x-size:x+size+1, y-size:y+size+1, 0].flatten()
      )
      Y.append(img[x, y, 1:])

  return X, Y

def rebuild(img, size=block_size):
  fig = plt.figure()
  plt.imshow(img)
  plt.xlabel('X axis')
  plt.ylabel('Y axis')
  plt.show()

  img = rgb2lab(img)
  w, h = img.shape[:2]
  
  print('Constructing window...')
  photo = np.zeros([w, h, 3])
  X = []
  for x in range(size, w - size):
    for y in range(size, h - size):
      window = img[x - size: x + size + 1, y - size: y + size + 1, 0].flatten()
      X.append(window)

  X = np.array(X)

  print('Predicting...')
  pred_ab = knn.predict(X).reshape(w - 2 * size, h - 2 * size, -1)
  photo[:, :, 0] = img[:, :, 0]
  photo[size: w - size, size: h - size, 1:] = pred_ab
  
  photo = photo[size: w - size, size: h - size, :]
  return photo


if __name__ == '__main__':
  X, Y = read_style_image(os.path.join(path, 'style.jpg'))
  knn = KNeighborsRegressor(n_neighbors=4, weights='distance')
  knn.fit(X, Y)
  
  content = io.imread(os.path.join(path, 'input.jpg'))
  new_photo = rebuild(content)
  new_photo = lab2rgb(new_photo)

  fig = plt.figure()
  plt.imshow(new_photo)
  plt.xlabel('X axis')
  plt.ylabel('Y axis')
  plt.show()