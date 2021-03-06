# Coat
---
__Coat__ is small wrapper that sits on __numpy__'s ndarray (*subclassed*) and __opencv__\
 Coat's only purpouse is for rapid prototyping

---
### Style

__Coat__ is using method cascading\
return from every method is __Coat__'s HigherCoating instance\
Only exception is method classic() which returns back numpy instance\

### Install

preferably use virtual env

```bash
pip install coat

```

### Demo

function __Coat__ is a proxy function that handles different types of args.

It handles str(url), list/generator of images and ndarray. 


```python
from coat import Coat

url = "https://natgeo.imgix.net/subjects/headers/shutterstock_276228476.jpg?auto=compress,format&w=1920&h=960&fit=crop"
Coat(url).thresh(125,255).show()

```
![](https://raw.githubusercontent.com/moyogatomi/coat/master/samples/earth_thresh.jpg)


## Usage

__Content__

[Core functionalities](#Core-functionalities)\
[Dominance](#Auto-resolver)\
[Array manipulation](#Array-manipulation)\
[Image downloading](#Download-image)\
[Labeling](#Supports-labeling)\
[Color spaces](#Colorspace-change)\
[Helper functions](#Image-processing-helpers)\
[Montage](#Montage)\
[Contours](#Contours)\
[Color filtering](#Replace-particular-color)\
[Motion difference](#Motion-difference)

---

### Core functionalities
Lets define two arrays of different type, size and dimension
```python
# Import Coat(proxy function) and Montage
from coat import Coat, Montage
import numpy as np

array1 = Coat(np.zeros(shape=[40,60,3],dtype=np.uint8))
array2 = Coat(np.zeros(shape=[8,8],dtype=np.float32))
```
##### Auto resolver
Coating the arrays you let __Coat__ resolve array operations whenever there is conflict 
```python
res = array1 + array2

res.shape,res.dtype
>>> (40, 60, 3) uint8
```
Coated arrays can be dominant or non-dominant.
General rules are as follows:
```sh
A(non-dominant) + B(non-dominant) -> A is prioritized
A(non-dominant) + B(dominant)     -> B is prioritized
A(dominant) + B(non-dominant)     -> A is prioritized
A(dominant) + B(dominant)         -> A is prioritized
```
##### Dominance setting
You can set which array is dominant
Be default each array is not dominant.
```python
res = array1 * array2.host()
res.shape,res.dtype
>>> (8, 8) float32
```
If have arrays of different size and dimension( gray & colored), you can simply transform to common standard as follows
```python
list_of_images = [img1,img2, img3, .., .., imgN]

# define common standard
grayscale_template = Coat(np.zeros(shape=[100,100],dtype=np.uint8))
colored_template = Coat(np.zeros(shape=[100,100],dtype=np.uint8))

grascaled = [grayscale_template.host() + image for image in list_of_images]
colored =   [colored_template.host() + image for image in list_of_images]
```
##### interpolation
You can set interpolation algorithm (default is LINEAR - BILINEAR)
```python
res = array1 - array2.host('CUBIC')
```

##### Removing dominance
```python
res = array1 * array2.host('CUBIC').guest()
res.shape,res.dtype
>>> (40, 60, 3) uint8
```
### Array manipulation

##### osize as objective size
```python
array1.shape, array2.shape
>>> (40, 60, 3) , (8,8)
array1.osize(array2.shape).shape
>>> (8,8,3)
array2.osize(array1.shape).shape
>>> (40, 60)
```

##### rsize as relative size
```python
res = array2.rsize(fx = 2, fy=0.5)
res.shape
>>> (16, 4)
```
###### ndarray compatibility
```python
res = array1 + np.zeros(shape=array1.shape)
np.uint8(array1) # -> returns back Coat instance with changed datatype
array1.classic() # -> returns back numpy instance
```

# Leveraging OpenCV

##### Download image
pass url to Coat and __show__ it
```python
url = "https://natgeo.imgix.net/subjects/headers/shutterstock_276228476.jpg?auto=compress,format&w=1920&h=960&fit=crop"
image = Coat(url).show()
image = image.rsize(fx=0.25,fy=0.25)
```
![](https://raw.githubusercontent.com/moyogatomi/coat/master/samples/earth.jpg)
##### Supports labeling
```python
# Use int for objective coordinates
image.labelbox("Home",(0,136),(230,340), bcolor = [0,255,0]).show()
# Use float for relative coordinates
image.labelbox("Home",(0.0,0.3),(0.9,0.7), bcolor = [0,255,0]).show()
```
![](https://raw.githubusercontent.com/moyogatomi/coat/master/samples/earth_label.jpg)

##### Colorspace change
supported color transformation
```
BGR2GRAY
BGR2HLS
BGR2HSV
HSV2BGR
GRAY2RGB
GRAY2BGR
RGB2GRAY
RGB2HSV
RGB2HLS
```
```python
image.color_to('BGR2GRAY')
```
##### NOTE
OpenCV is using BGR as default color scheme
##### Image processing helpers
Threshold
```python
image.thresh(125,255,'thresh_binary').show()
```
![](https://raw.githubusercontent.com/moyogatomi/coat/master/samples/earth_thresh.jpg)

Convolution filtering
```python
image.blur_median
image.filter_bilateral
image.blur_gauss
image.blur_average
```
Convolution 2D
```python
img.conv(kernel)
```

##### Montage

See orignal next to processed image\
we add host (turn on dominance of first image) so we get result in RGB colorspace as our original image is rgb
```python
image.rsize(fx=0.3,fy=0.3).host().join(image.thresh(127,255)).show()
```
![](https://raw.githubusercontent.com/moyogatomi/coat/master/samples/earth_thresh_join.jpg)
Montage of different color spaces
```python
img = image.rsize(fx=0.3,fy=0.3)
color_spaces = ["BGR2GRAY","BGR2HLS","BGR2HSV",]
```
Dominant is the first image if template is not defined
```python
all_images = [img] + [img.to_color(cspace) for cspace in color_spaces]
montage = Montage(all_images).grid(2,2).show()
```
![](https://raw.githubusercontent.com/moyogatomi/coat/master/samples/montage_colorspaces.jpg)

resize montage based on template
```python
montage = Montage(all_images).template(np.zeros(shape=[50,50,3],dtype=np.uint8)).grid(2,2)
```

##### Remove stars with morphological opening
```python
image.morphologyEx('open',3).show()
```
![](https://raw.githubusercontent.com/moyogatomi/coat/master/samples/no_stars.jpg)

##### Contours
```python
# Draw quick countours
thr = image.thresh(200,255)

# copy 
contoured = image.copy().contours(thr,min_size=5, max_size = 9999999,thickness=2,color = [0,125,255]).show()
```
![](https://raw.githubusercontent.com/moyogatomi/coat/master/samples/contour1.jpg)

##### Replace particular color
```python
present = [0,125,255]
future = [255,0,0]
contoured.replace(present,future).show()
```
![](https://raw.githubusercontent.com/moyogatomi/coat/master/samples/contour2.jpg)
##### Color filtering
```python
# [36,0,0] --> green color interval in HSV <--[70,255,255]
image.filterHsv([36,0,0],[70,255,255],passband=True).show()
image.filterHsv([36,0,0],[70,255,255],passband=False).show()
# Passband False:   -------|++|----- 
# Passband True:   ++++++|--|++++++ 
```
passband True

![](https://raw.githubusercontent.com/moyogatomi/coat/master/samples/filter_true.jpg)


passband False

![](https://raw.githubusercontent.com/moyogatomi/coat/master/samples/filter_false.jpg)

##### Motion difference
```python
box1 = Coat(np.zeros(shape=[400,400,3])).box((30,30),(250,250),color=[255,125,0])
box2 = Coat(np.zeros(shape=[400,400,3])).box((30,150),(250,350),color=[255,125,0])

motion_diff = box1.motion_difference(box2,val=30).show()
```
![](https://raw.githubusercontent.com/moyogatomi/coat/master/samples/mdiff.jpg)

