import matplotlib.pyplot as plt
from quiver import compute_gradient
from coat import Coat
import numpy as np
from scipy.signal import savgol_filter

img_url = 'test.png'
img = Coat(img_url).rsize(fx=1,fy=1).to_color('BGR2GRAY').astype(np.float32)/255.
#img[0:60,-80:-20].show()

Jx,Jy,G,tan,atan = compute_gradient(img,gradient_operator=[1,1],smooth=True,gauss_kernel=9)

# grid_G.show()
# grid.show()
# x = grid[np.where(grid_G>0.01)]

# g = np.histogram(x,bins=36)
# print(g)
# plt.plot(list(range(0,360,360//36)),g[0])
# plt.show()

class KeyDescriptor:
    def __init__(self,full_grid,G,tan,threshold=0.1,bins=36):
        self.full_grid = full_grid
        self.threshold = threshold
        self.bins = bins
        self.tan = tan
        self.G= G

    def distribution(patch_size,sigma = 1):
        x,y = np.meshgrid(np.linspace(-1,1,patch_size),np.linspace(-1,1,patch_size))
        sigma = 1
        dist = np.exp(-((x**2 + y**2)/(2.0*sigma**2))) / (2*np.pi*sigma**2)
        return dist/dist.max()

    def key(self):
        x = self.tan[np.where(self.G>self.threshold)]
        g = np.histogram(x,bins=self.bins)[0]
        return g
    def _key(self,tan,G):
        x = tan[np.where(G>self.threshold)]
        g = np.histogram(x,bins=self.bins)[0]
        return g
    def quadrants(self,tan,G,q=4):
        q = int(np.sqrt(q))
        qx_m = G.shape[0]//q
        qy_m = G.shape[1]//q
        quads = []
        for qx in range(q):
            for qy in range(q):
                quadrant = self._key(tan[qx*qx_m:(qx+1)*qx_m,qy*qy_m:(qy+1)*qy_m],G[qx*qx_m:(qx+1)*qx_m,qy*qy_m:(qy+1)*qy_m])
                quads.append(quadrant)
        return quads
from scipy import ndimage
print(img.shape)
img1 = img[120:200,120:200]
img_45 = ndimage.rotate(img.copy(), 23, reshape=False,order=1)
img_45 = img_45/img_45.max()
img_45[img_45<0] = 0
Coat(img_45)[120:200,120:200]

bins = 36
Jx,Jy,G,tan,atan = compute_gradient(img,gradient_operator=[1,1],smooth=True,gauss_kernel=3)
K1 = KeyDescriptor(None,G[120:200,120:200],tan[120:200,120:200].show(),bins=bins)
print(G.max(),G.min())
print(tan.max(),tan.min())

#tan.show()
Jx,Jy,G,tan,atan = compute_gradient(Coat(img_45),gradient_operator=[1,1],smooth=True,gauss_kernel=3)
K2 = KeyDescriptor(None,G[120:200,120:200],tan[120:200,120:200].show(),bins=bins)
#tan.show()
# a = KeyDescriptor(None,tan[0:60,0:60],G[0:60,0:60].show(),bins=bins)
print(G.max(),G.min())
print(tan.max(),tan.min())
def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w
#savgol_filter(y, 51, 3) # window size 51, polynomial order 3
K1_v = moving_average(K1.key(),1)
K2_v = moving_average(K2.key(),1)

bins = K1_v.shape[0]
normalize = True
if normalize:
    K1_cor = (K1_v - np.mean(K1_v)) / (np.std(K1_v) * K1_v.shape[0])
    K2_cor = (K2_v - np.mean(K2_v)) /  np.std(K2_v)
cross_correlation = np.correlate(K1_cor,K2_cor, "full")


x_cross_corr = np.linspace(-360,360,cross_correlation.shape[0])
phase = x_cross_corr[np.argmax(cross_correlation)]


phase_to_index = int(phase*bins/360)
print(phase)

plt.plot(np.linspace(0,360,bins),K1_v,'r')
plt.plot(np.linspace(0,360,bins),np.roll(K2_v,int(phase_to_index)),'b')
#plt.plot(x_cross_corr,cross_correlation*90)
plt.show()
print(np.corrcoef(K1_v,K2_v))




