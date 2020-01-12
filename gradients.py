from coat import Coat
from coat import Montage
import numpy as np
import cv2


def grad_kernel(core):
    core = np.array(core)
    kernelx = np.zeros(shape=[core.shape[0],core.shape[0]])
    kernelx[:,0] = -core
    kernelx[:,-1]= core

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
    magnitude = np.sqrt(np.float32(Jx)**2+np.float32(Jy)**2)
    phase=cv2.phase(Jx,Jy,True)/(2*np.pi)
    atan = np.arctan(Jx,Jy)
    #print(Coat(tan).osize((5,5)))
    #exit()
    return Coat(Jx),Coat(Jy),Coat(magnitude),Coat(phase),atan

def quiver(image,fraction=10,scale=500000):
    Jx,Jy,G,tan,atan = Coat(compute_gradient(image,gradient_operator=[1,1],smooth=True))
    Jx,Jy = Coat(np.gradient(image))
    G = np.sqrt(Jx**2+Jy**2)

    K = Jy/Jx
    shape = image.shape
    for x in range(shape[0]//fraction):
        x_pos = x*fraction
        for y in range(shape[1]//fraction):
            y_pos = y*fraction
            M = G[x,y]*scale
            Kt = K[x,y]
            print(M,Kt)
            try:
                x_pos2 =x_pos+int(np.sqrt(M*M/(1+Kt*Kt)))
                y_pos2 =y_pos+int(np.sqrt(M*M/(1+(1/(Kt*Kt)))))
                print(x_pos,x_pos2)
                cv2.arrowedLine(image, (x_pos,y_pos), (x_pos2,y_pos2), 0, 1)
            except Exception as e:
                print(e)
    #exit()
    return image


img = Coat('https://i.stack.imgur.com/OZuBo.png').rsize(fx=0.5,fy=0.5).astype(np.float32)/255.
#img = Coat('gimage.png').rsize(fx=1,fy=1).astype(np.float32)/255.
quiver(img.to_color('BGR2GRAY'),fraction=50,scale=200).show()
#exit()
# # kernely = np.array([[1,2,1],[0,0,0],[-1,-2,-1]])
# # kernelx = np.array([[1,2,-1],[1,0,-1],[1,0,-1]])
# # edges_x = cv2.filter2D(img,cv2.CV_8U,kernelx)
# # edges_y = cv2.filter2D(img,cv2.CV_8U,kernely)

# # sobel = np.hypot(edges_x**2,edges_y**2)
# # print(sobel)

# # Coat(sobel.astype(np.uint8)).show()


Jx,Jy,M,tan,atan = compute_gradient(img.to_color('BGR2GRAY'),gradient_operator=[1,1],smooth=True)

(Jx/2+0.5).show()
(Jy/2+0.5).show()

print(tan.max(),tan.min())
M.show()
tan.show()
exit()
# #cv2.waitKey(0)


# exit()
















# image = Coat('gimage.png').to_color('BGR2GRAY')/255

# # x,y = np.gradient(image)
# # Coat(x).show()
# # Coat(y).show()

# #g = Coat(compute_gradient(image,gradient_operator=[1,1,1,1])[-1])

# size = 300
# d = np.linspace(0,1,size).reshape(1,-1)
# triangle = Coat(1-np.dot(d.T,d)).show()

# Jx,Jy,G,tan = Coat(compute_gradient(triangle,gradient_operator=[1,1],smooth=False))


# import matplotlib.pyplot as plt
# import numpy as np

# X = np.arange(size)
# Y = np.arange(size)
# U, V = np.meshgrid(X, Y)

# # #x,y = np.meshgrid(np.arange(-2, 2, .2), np.arange(-2, 2, .25))
# # print(U.shape,V.shape,Jx.shape,Jy.shape)
# # fig, ax = plt.subplots()
# # q = ax.quiver(U,V,Jx,Jy)
# # plt.show()
# def quiver(image,fraction=50,scale=500000):
#     Jx,Jy,G,tan = Coat(compute_gradient(image,gradient_operator=[1,1,1,1],smooth=False))
#     K = Jy/Jx
#     shape = image.shape


#     for x in range(shape[0]//fraction):
#         x_pos = x*fraction
#         for y in range(shape[1]//fraction):
#             y_pos = y*fraction
#             M = G[x,y]*scale
#             Kt = K[x,y]
#             try:
#                 x_pos2 =x_pos+int(np.sqrt(M*M/(1+Kt*Kt)))
#                 y_pos2 =y_pos+int(np.sqrt(M*M/(1+(1/(Kt*Kt)))))
#                 print(x_pos,x_pos2)
#                 cv2.arrowedLine(image, (x_pos,y_pos), (x_pos2,y_pos2), 0, 2)
#             except:
#                 pass
#     return image

# quiver(triangle,fraction=50).show()
# exit()
# triangle.show()



import matplotlib.pyplot as plt
import numpy as np

fig, ax = plt.subplots()
Jx,Jy = Coat(np.gradient(img.to_color('BGR2GRAY')))
# q = ax.quiver(Jx.classic()[1:-1,:-2],-Jy.classic()[:-2,1:-1],color='r')
# r = ax.quiver(Jx.classic()[1:-1,:-2],Jy.classic()[:-2,1:-1],color='b')
# s = ax.quiver(-Jx.classic()[1:-1,:-2],-Jy.classic()[:-2,1:-1],color='g')

q = ax.quiver(Jx.classic(),-Jy.classic(),color='r')
r = ax.quiver(Jx.classic(),Jy.classic(),color='b')
s = ax.quiver(-Jx.classic(),-Jy.classic(),color='g')
print((Jy/Jx))
plt.imshow(img,cmap='gray', vmin=0, vmax=1)
plt.show()
