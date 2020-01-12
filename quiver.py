import matplotlib.pyplot as plt
import numpy as np
import cv2
from coat import Coat

def grad_kernel(core):
    core = np.array(core)
    kernelx = np.zeros(shape=[core.shape[0],core.shape[0]])
    kernelx[:,0] = -core
    kernelx[:,-1]= core
    if kernelx.sum()!=0:
        kernelx = kernelx/kernelx.sum()
    return kernelx,kernelx.T

def compute_gradient(image,gradient_operator=None, gauss_kernel=5, smooth = True):
    if gradient_operator is None:
        gradient_operator = [1,1]
    if image.dtype == 'uint8':
        conv_type = cv2.CV_8U
    else:
        conv_type = cv2.CV_32F
    kx,ky = grad_kernel(gradient_operator)
    Jx = cv2.filter2D(image.copy().blur_gauss((gauss_kernel,gauss_kernel)),conv_type,kx)if smooth else cv2.filter2D(image.copy(),conv_type,kx)
    Jy = cv2.filter2D(image.copy().blur_gauss((gauss_kernel,gauss_kernel)),conv_type,ky)if smooth else cv2.filter2D(image.copy(),conv_type,ky)
    #Jx = cv2.Sobel(image,cv2.CV_32F,1,0,ksize=9)
    #Jy = cv2.Sobel(image,cv2.CV_32F,0,1,ksize=9)
    magnitude = np.sqrt(np.float32(Jx)**2+np.float32(Jy)**2)
    phase=cv2.phase(Jx,Jy,True)/(2*np.pi)
    atan = (np.pi+np.arctan2(Jx,Jy))/(2*np.pi)

    #exit()
    #print(Coat(tan).osize((5,5)))
    #exit()
    return Coat(Jx),Coat(Jy),Coat(magnitude),Coat(phase),Coat(atan)

def phase2k(p):
    p = p*360
    return p*np.pi/180

def new_positions(pos: tuple,phase: float, magnitude: float):
    x_pos, y_pos = pos
    M = magnitude
    x_pos2 =x_pos+int(np.sin(phase)*M)
    y_pos2 =y_pos+int(np.cos(phase)*M)
    return x_pos, y_pos, x_pos2, y_pos2


def quiver(image,fraction=10,scale=5):
    Jx,Jy,G,tan,atan = compute_gradient(image,gradient_operator=[1,1,1,1,1],smooth=True,gauss_kernel=3)
    tan.show()
    new_image = (image.copy()*255).astype(np.uint8).to_color('GRAY2BGR')
    counter = 0
    K = Jy/Jx
    shape = image.shape
    for x in range(shape[0]//fraction):
        x_pos = x*fraction
        for y in range(shape[1]//fraction):
            y_pos = y*fraction
            #new_image[x_pos,y_pos]=[0,255,0]
            searched_value = np.round(G[x_pos:x_pos+fraction,y_pos:y_pos+fraction],3).max()
            ind = np.where(np.round(G[x_pos:x_pos+fraction,y_pos:y_pos+fraction],3)==searched_value)

            x_poss,y_poss = x_pos + ind[0][0],y_pos + ind[1][0]


            M = G[x_poss,y_poss]*scale
            Kt = tan[x_poss,y_poss]


            try:

                x_poss, y_poss, x_pos2, y_pos2 = new_positions((x_poss,y_poss),phase2k(Kt+0.75),M)
                x_poss_t, y_poss_t, x_pos2_t, y_pos2_t = new_positions((x_poss,y_poss),phase2k(Kt+0.5),M)
                cv2.arrowedLine(new_image, (y_poss,x_poss), (y_pos2,x_pos2), (0,255,0), 1)
                cv2.arrowedLine(new_image, (y_poss_t,x_poss_t), (y_pos2_t,x_pos2_t), (0,0,255), 1)
            except Exception as e:
                print(e)
    return new_image
if __name__ == '__main__':

    from coat import Coat
    import cv2





    img_url = 'https://i.stack.imgur.com/OZuBo.png'
    img_url = 'gimage.png'
    image = Coat(img_url).to_color('BGR2GRAY').rsize(fx=1,fy=1).astype(np.float32)/255.
    # image.show()
    # #Jx,Jy = np.gradient(image,edge_order=1)

    quiver(image,fraction=3,scale=5).show()
    # fig, ax = plt.subplots()

    # s = ax.quiver(Jx.classic(),Jy.classic(),color='g')

    # plt.imshow(image,cmap='gray', vmin=0, vmax=1)
    # plt.show()
