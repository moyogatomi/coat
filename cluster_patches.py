from coat import Coat
import time
from sklearn.feature_extraction import image
from coat import Montage


import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from sklearn.cluster import KMeans
img_url = 'https://i.stack.imgur.com/OZuBo.png'
img_url = 'gimage.png'

IMAGE = Coat(img_url).rsize(fx=0.5,fy=0.5)#.to_color('BGR2GRAY')


def segmentation(img,size,clusters=10,dist=False):
    s = size
    x = image.extract_patches_2d(IMAGE, (s, s))
    data = x.reshape(x.shape[0],-1)
    extra = np.zeros(shape=[data.shape[0],2])
    counter = 0
    for i in range(IMAGE.shape[0]-s+1):
        for j in range(IMAGE.shape[1]-s+1):
            extra[counter]=[i/IMAGE.shape[0],j/IMAGE.shape[1]]
            counter +=1
    if dist:
        data = (np.hstack((data,extra)))
    K = clusters
    kmeans5 = KMeans(n_clusters=K)
    #data = np.concatenate(b, axis=0)
    #data -= np.mean(data, axis=0)
    #data /= np.std(data, axis=0)
    start = time.time()

    print('starting')
    y_kmeans5 = kmeans5.fit_predict(data)
    print('finishing',time.time()-start)
    a = np.zeros(shape=IMAGE.shape,dtype=np.float32)
    counter = 0
    for x in tqdm(range(IMAGE.shape[0]-s+1)):
        for y in range(IMAGE.shape[1]-s+1):
            a[x:x+s,y:y+s]=y_kmeans5[counter]
            counter +=1
    return Coat(a/K)
img1 = segmentation(IMAGE,3,clusters=5,dist=True)
img2 = segmentation(IMAGE,3,clusters=5 ,dist=False)
Montage([img1,img2]).grid(1,2).show()
