import matplotlib.pyplot as plt
from quiver import compute_gradient
from coat import Coat
import numpy as np
from scipy.signal import savgol_filter



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
    
    def moving_average(self,x, w):
        return np.convolve(x, np.ones(w), 'valid') / w

    def normalize_superion(self,K):
        return (K - np.mean(K)) / (np.std(K) * K.shape[0])

    def normalize_inferion(self,K):
        return (K - np.mean(K)) /  np.std(K)

    def cross_correlation(self, K_S, K_I):
        cross_correlation = np.correlate(K_S,K_I, "full")
        x_cross_corr = np.linspace(-360,360,cross_correlation.shape[0])
        return cross_correlation, x_cross_corr

    def phase(self,x_cross_corr,cross_correlation):
        phase = x_cross_corr[np.argmax(cross_correlation)]
        phase_to_shift = int(phase*cross_correlation.shape[0]*0.5/360) #bins = K_S.shape[0]
        # plt.plot(np.linspace(0,360,bins),K1_v,'r')
        # plt.plot(np.linspace(0,360,bins),np.roll(K2_v,int(phase_to_index)),'b')
        # plt.show()
        return phase, phase_to_shift

def normalize(K1,K2):
    K1_cor = (K1 - np.mean(K1)) / (np.std(K1) * K1.shape[0])
    K2_cor = (K2 - np.mean(K2)) /  np.std(K2)
    return K1_cor, K2_cor

    

from scipy import ndimage

img_url = 'test.png'
bins = 360

img = Coat(img_url).rsize(fx=1,fy=1).to_color('BGR2GRAY').astype(np.float32)/255.
img_45 = ndimage.rotate(img.copy(), 45, reshape=False,order=1) 
img_45 = img_45/img_45.max()
img_45[img_45<0] = 0


from tqdm import tqdm
import time
s = time.time()
for i in (range(200)):
    Jx,Jy,G,tan,atan = compute_gradient(Coat(img_45),gradient_operator=[1,1],smooth=True,gauss_kernel=3)
    K1 = KeyDescriptor(None,G[120:220,120:200],tan[120:200,120:200],bins=bins)
    Jx,Jy,G,tan,atan = compute_gradient(img,gradient_operator=[1,1],smooth=True,gauss_kernel=3)

    K2 = KeyDescriptor(None,G[120:200,120:200],tan[120:200,120:200],bins=bins)

    def moving_average(x, w):
        return np.convolve(x, np.ones(w), 'valid') / w


    K1_v = moving_average(K1.key(),1)
    K2_v = moving_average(K2.key(),1)

    bins = K1_v.shape[0]


    K1_cor , K2_cor =  normalize(K1_v,K2_v)

    cross_correlation = np.correlate(K1_cor,K2_cor, "full")
    x_cross_corr = np.linspace(-360,360,cross_correlation.shape[0])

    phase = x_cross_corr[np.argmax(cross_correlation)]


    phase_to_index = int(phase*bins/360)

    # plt.plot(np.linspace(0,360,bins),K1_v,'r')
    # plt.plot(np.linspace(0,360,bins),np.roll(K2_v,int(phase_to_index)),'b')
    # plt.show()

print(time.time()-s)
    #print(np.corrcoef(K1_cor,np.roll(K2_v,int(phase_to_index))))




