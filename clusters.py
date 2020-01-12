from coat import Coat
import time


import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from sklearn.cluster import KMeans
img_url = 'https://i.stack.imgur.com/OZuBo.png'
img_url = 'gimage.png'
x = Coat(img_url).rsize(fx=0.3,fy=0.3)

b = []
s = 10

for i in tqdm(range(x.shape[0]-s)):
    for j in range(x.shape[1]-s):
        b.append(x[i:i+s,j:j+s].reshape(1,-1))

K = 10
kmeans5 = KMeans(n_clusters=K)
data = np.concatenate(b, axis=0)
#data -= np.mean(data, axis=0)
#data /= np.std(data, axis=0)
start = time.time()

print('starting')
y_kmeans5 = kmeans5.fit_predict(data)
print('finishing',time.time()-start)
a = np.zeros(shape=x.shape)
counter = 0
for i in tqdm(range(x.shape[0]-s)):
    for j in range(x.shape[1]-s):
        a[i:i+s,j:j+s]=y_kmeans5[counter]
        counter +=1
Coat(x).show()
Coat(a/K).show()