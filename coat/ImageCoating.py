#from __future__ import annotations
from typing import Union
import inspect
import cv2
import numpy as np
import urllib
import requests
import validators
from functools import wraps
from coat.helpers import create_label, urlvalidation, url_to_image
from PIL import Image
from types import GeneratorType

def copy_meta(func):
    """Summary
    
    Args:
        func (TYPE): Description
    
    Returns:
        TYPE: Description
    """
    @wraps(func)
    def wrapper(self,*args,**kwargs):
        obj = func(self,*args,**kwargs)
        obj.__dict__ = self.__dict__
        return obj
    return wrapper

def Coat(array, dtype=None):
    """function to decide what to do with input
    
    Parameters
    ----------
    array : ndarray, generator, list, str
        anything that represents an image or container of images. Including str representing url to image
    
    dtype : None, optional
        output datatype. Use np datatypes (np.uint8, np.float32)
    
    Returns
    -------
    Coating
        ndarray wrapper or generator of ndarray wrappers
    """

    if isinstance(array,Image.Image):
        array = np.array(array)
        if len(array.shape)==3:
            array= array[:,:,::-1]

    if isinstance(array, np.ndarray):
        if dtype is None:
            dtype = array.dtype
        template = (array.view(HigherCoating)).astype(dtype)
        return template

    if isinstance(array, (list, tuple, GeneratorType)):
        iterable = IterableCoating()
        return iterable(array, dtype)

    if isinstance(array, str):
        if urlvalidation(array):
            image = url_to_image(array)
            template = (image.view(HigherCoating)).astype(np.uint8)
            return template
        else:
            if array.lower().endswith(".npy"):
                return Coat(np.load(array))
            if array.lower().endswith(
                (".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".gif")
            ):
                return Coat(cv2.imread(array, cv2.IMREAD_COLOR))


class IterableCoating:
    def __call__(self, iterable, dtype):
        for item in iterable:
            yield item.view(HigherCoating).astype(dtype)


class Coating(np.ndarray):
    dominant = False
    
    colorization = {3: cv2.COLOR_GRAY2BGR, 2: cv2.COLOR_BGR2GRAY}
    interpolation = {
        "LINEAR": cv2.INTER_LINEAR,
        "CUBIC": cv2.INTER_CUBIC,
        "NEAREST": cv2.INTER_NEAREST,
    }
    inter = interpolation["LINEAR"]

    def __init__(self,*args,**kwargs):
        """Summary
        
        Args:
            *args: Description
            **kwargs: Description
        """
        self.dominant = False

    def __new__(cls, *args, **kwargs):
        instance = super().__new__(cls, *args, **kwargs)
        return instance

    def __array_finalize__(self, obj):
        
        """
        Pass variables whenever using numpy on array
        """

        if isinstance(obj, (Coating, HigherCoating)):
            if inspect.getouterframes( inspect.currentframe())[1].function=='<module>':
                self.__dict__ = obj.__dict__

    def __init__(self, *args, **kwargs):
        if len(self.shape) < 2:
            raise Exception("Not accustomed for 1D")

    def __call__(self, array):
        return Coat(array, dtype=np.dtype(array.dtype))

    def classic(self):
        """ transform back to numpy 
        
        Returns:
            ndarray: Returns back pure ndarray 
        """
        return np.array(self)

    def _copy_dict(self, obj):
        obj.__dict__ = self.__dict__
        return obj

    def pil(self,flip_channels=True):
        if flip_channels and len(self.shape)==3:
            return Image.fromarray(np.uint8(self[:,:,::-1])) 
        return Image.fromarray(np.uint8(self)) 


    def blend(self, _self, _other):
        """ Auto resolving functions

        this functions blends size, dimension and datatype.
        Prioritized array is the host() one
        
        Args:
            _self (TYPE): self
            _other (TYPE): other
        
        Returns:
            new instance: 
        """

        if isinstance(_other, np.ndarray) and isinstance(_self, np.ndarray):
            other = _other.copy()            
            if _other.dtype == np.float16:
                _other = _other.astype(np.float32)
            if _self.dtype == np.float16:
                _self = _self.astype(np.float32)

            if _self.shape[:2] != _other.shape[:2]:
                if hasattr(_other, "dominant"):
                    if _self.dominant:
                        _other = Coat(
                            cv2.resize(
                                _other, _self.shape[:2][::-1], interpolation=self.inter
                            )
                        )
                    elif not _self.dominant and _other.dominant:
                        _self = Coat(
                            cv2.resize(
                                _self, _other.shape[:2][::-1], interpolation=self.inter
                            )
                        )
                    else:
                        _other = Coat(
                            cv2.resize(
                                _other, _self.shape[:2][::-1], interpolation=self.inter
                            )
                        )
                else:
                    _other = Coat(
                        cv2.resize(
                            _other, _self.shape[:2][::-1], interpolation=self.inter
                        )
                    )

            if _self.shape != _other.shape:
                if _self.dominant:
                    _other = Coat(
                        cv2.cvtColor(
                            np.uint8(_other), _self.colorization[len(_self.shape)]
                        )
                    )
                elif not _self.dominant and _other.dominant:
                    _self = Coat(
                        cv2.cvtColor(
                            np.uint8(_self), _self.colorization[len(_other.shape)]
                        )
                    )
                else:
                    _other = Coat(
                        cv2.cvtColor(
                            np.uint8(_other), _self.colorization[len(_self.shape)]
                        )
                    )

            #return (_self), other._copy_dict(_other) if isinstance(other,self.__class__) else Coat(other)._copy_dict(_other)
        return _self, _other

    def host(self, *args):
        """ Make instance prioritized
        
        Args:
            *args: str representing interpolation algorithm
        
        Returns:
            self: self
        """
        if args:
            interp = self.interpolation.get(args[0], None)
            if interp is not None:
                self.inter = interp

        self.dominant = True
        return self

    def guest(self):
        """ Set array as non-dominant, which is already defualt value
        
        Returns:
            self: self
        """
        self.dominant = False
        return self

    #@copy_meta
    def osize(self, size: tuple) -> 'HigherCoating':
        """Objective resize
        
        Args:
            size (tuple): tuple of ints
        
        Returns:
            self: self
        """
        return Coat(cv2.resize(self, size[:2][::-1]))

    #@copy_meta
    def rsize(self, fx: float, fy: float) -> 'HigherCoating':
        """Summary
        
        Args:
            fx (Union): Description
            fy (TYPE): Description
        
        Returns:
            HigherCoating: Description
        """
        return Coat(cv2.resize(self, None, fx=fy, fy=fx))

    def __add__(self, other):
        self, other = self.blend(self, other)
        return super().__add__(other)

    def __sub__(self, other):
        self, other = self.blend(self, other)
        return super().__sub__(other)

    def __mul__(self, other):
        self, other = self.blend(self, other)
        return super().__mul__(other)

    def __floordiv__(self, other):
        self, other = self.blend(self, other)
        return super().__floordiv__(other)

    def __truediv__(self, other):
        self, other = self.blend(self, other)
        return super().__truediv__(other)

    def __mod__(self, other):
        self, other = self.blend(self, other)
        return super().__mod__(other)

    def __lshift__(self, other):
        self, other = self.blend(self, other)
        return super().__lshift__(other)

    def __rshift__(self, other):
        self, other = self.blend(self, other)
        return super().__rshift__(other)

    def __and__(self, other):
        self, other = self.blend(self, other)
        return super().__and__(other)

    def __xor__(self, other):
        self, other = self.blend(self, other)
        return super().__xor__(other)

    def __or__(self, other):
        self, other = self.blend(self, other)
        return super().__or__(other)

    def __iadd__(self, other):
        self, other = self.blend(self, other)
        return super().__iadd__(other)

    def __isub__(self, other):
        self, other = self.blend(self, other)
        return super().__isub__(other)

    def __imul__(self, other):
        self, other = self.blend(self, other)
        return super().__imul__(other)

    def __idiv__(self, other):
        self, other = self.blend(self, other)
        return super().__idiv__(other)

    def __ifloordiv__(self, other):
        self, other = self.blend(self, other)
        return super().__ifloordiv__(other)

    def __imod__(self, other):
        self, other = self.blend(self, other)
        return super().__imod__(other)

    def __ilshift__(self, other):
        self, other = self.blend(self, other)
        return super().__ilshift__(other)

    def __irshift__(self, other):
        self, other = self.blend(self, other)
        return super().__irshift__(other)

    def __iand__(self, other):
        self, other = self.blend(self, other)
        return super().__iand__(other)

    def __ixor__(self, other):
        self, other = self.blend(self, other)
        return super().__ixor__(other)

    def __ior__(self, other):
        self, other = self.blend(self, other)
        return super().__ior__(other)

    def __lt__(self, other):
        self, other = self.blend(self, other)
        return super().__lt__(other)

    def __le__(self, other):
        self, other = self.blend(self, other)
        return super().__le__(other)

    def __eq__(self, other):
        self, other = self.blend(self, other)
        return super().__eq__(other)

    def __ne__(self, other):
        self, other = self.blend(self, other)
        return super().__ne__(other)

    def __ge__(self, other):
        self, other = self.blend(self, other)
        return super().__ge__(other)

    def __gt__(self, other):
        self, other = self.blend(self, other)
        return super().__gt__(other)


class HigherCoating(Coating):
    MORPH = {"close": cv2.MORPH_CLOSE, "open": cv2.MORPH_OPEN}
    THRESH = {
        "thresh_binary": cv2.THRESH_BINARY,
        "thresh_binary_inv": cv2.THRESH_BINARY_INV,
        "thresh_trunc": cv2.THRESH_TRUNC,
        "thresh_tozero": cv2.THRESH_TOZERO,
        "thresh_tozero_inv": cv2.THRESH_TOZERO_INV,
    }
    COLOR_SPACE = {
        "BGR2GRAY": cv2.COLOR_BGR2GRAY,
        "BGR2HLS": cv2.COLOR_BGR2HLS,
        "BGR2HSV": cv2.COLOR_BGR2HSV,
        "HSV2BGR": cv2.COLOR_HSV2BGR,
        "GRAY2RGB": cv2.COLOR_GRAY2RGB,
        "GRAY2BGR": cv2.COLOR_GRAY2BGR,
        "RGB2GRAY": cv2.COLOR_RGB2GRAY,
        "RGB2HSV": cv2.COLOR_RGB2HSV,
        "RGB2HLS": cv2.COLOR_RGB2HLS,
    }

    FONT = {
        "FONT_HERSHEY_COMPLEX": cv2.FONT_HERSHEY_COMPLEX,
        "FONT_HERSHEY_COMPLEX_SMALL": cv2.FONT_HERSHEY_COMPLEX_SMALL,
        "FONT_HERSHEY_DUPLEX": cv2.FONT_HERSHEY_DUPLEX,
        "FONT_HERSHEY_PLAIN": cv2.FONT_HERSHEY_PLAIN,
        "FONT_HERSHEY_SCRIPT_COMPLEX": cv2.FONT_HERSHEY_SCRIPT_COMPLEX,
        "FONT_HERSHEY_SCRIPT_SIMPLEX": cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
        "FONT_HERSHEY_SIMPLEX": cv2.FONT_HERSHEY_SIMPLEX,
        "FONT_HERSHEY_TRIPLEX": cv2.FONT_HERSHEY_TRIPLEX,
        "FONT_ITALIC": cv2.FONT_ITALIC,
    }

    KERNEL = dict(
        LAPLACIAN=np.array(([0, 1, 0], [1, -4, 1], [0, 1, 0]), dtype="int"),
        SOBELX=np.array(([-1, 0, 1], [-2, 0, 2], [-1, 0, 1]), dtype="int"),
        SOBELY=np.array(([-1, -2, -1], [0, 0, 0], [1, 2, 1]), dtype="int"),
        SHARPEN=np.array(([0, -1, 0], [-1, 5, -1], [0, -1, 0]), dtype="int"),
    )

    def __new__(cls, *args, **kwargs):
        instance = super().__new__(cls, *args, **kwargs)
        return instance

    #@copy_meta
    def morphologyEx(self, morph: str, kernel: int) -> 'HigherCoating':
        """ morpology
        
        Args:
            morph (str): close or open
            kernel (int): kernel size. eg: kernel = 3 -> np.ones(3,3)
        
        Returns:
            HigherCoating: Description
        """
        return Coat(
            cv2.morphologyEx(
                self, self.MORPH[morph], np.ones((kernel, kernel), np.uint8)
            )
        )

    def motion_difference(
        self, array, val=15, max_val=255, dtype=np.float16, output=np.uint8
    ) -> 'HigherCoating':
        """Summary
        
        Args:
            array (TYPE): Description
            val (int, optional): Threshold difference. val = 15 means that pixels 100 and 116 are considered to have movement
            max_val (int, optional): Description
            dtype (np.type, optional): For substracting we need signed float, so there is no overflow
            output (np.type, optional): As we work with images, uint8 is default value

        Returns:
            HigherCoating: Description
        """
        array = array.guest()  # Is it required here?

        mask = np.zeros(shape=self.shape[:2], dtype=np.uint8)
        if len(self.shape) == 3:
            diff = np.abs(dtype(self) - dtype(array)).sum(axis=2) / 3
        else:
            diff = np.abs(dtype(self) - dtype(array))

        mask[diff >= val] = 255
        mask[diff < val] = 0
        return Coat(mask)

    #@copy_meta
    def thresh(self, min_val: int, max_val: int, thresh_type="thresh_binary") -> 'HigherCoating':
        """ Threshold values
        
        Args:
            min_val (int): threshold value
            max_val (int): assigned value after triggering threshold
            thresh_type (str, optional): Description
        
        Returns:
            HigherCoating: Description
        """
        shape = self.shape
        if len(shape) == 2:
            ret, thr = cv2.threshold(self, min_val, max_val, self.THRESH[thresh_type])
        if len(shape) == 3:
            ret, thr = cv2.threshold(
                np.mean(self, axis=2), min_val, max_val, self.THRESH[thresh_type]
            )

        return Coat(np.uint8(thr))


    def contours_separate(
        self, array, min_size=10, max_size=10000, color=[255, 0, 0], thickness=-1
    ):
        """Summary
        
        Args:
            array (thresh): Threshed array that will serve for drawing counters
            min_size (int, optional): minimal size of countour to be drawn
            max_size (int, optional): maxiaml size of countour to be drawn
            color (list or int, optional): color of countour
            thickness (int, optional): thickness of contour. -1 fills all

        Returns:
            cavnas (RGB image): returns black image with contours
        """
        contours, _ = cv2.findContours(
            image=array, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE
        )
        canvas = np.uint8(np.zeros(shape=[array.shape[0], array.shape[1], 3]))

        cnts = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > min_size and area < max_size:
                cnts.append(cnt)
        cv2.drawContours(
            image=canvas, contours=cnts, contourIdx=-1, color=color, thickness=thickness
        )

        return Coat(canvas)

    #@copy_meta
    def contours(
        self, array, min_size=10, max_size=10000, color=[255, 0, 0], thickness=-1
    ):
        contours, _ = cv2.findContours(
            image=array, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE
        )

        cnts = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > min_size and area < max_size:
                cnts.append(cnt)
        cv2.drawContours(
            image=self, contours=cnts, contourIdx=-1, color=color, thickness=thickness
        )
        """Summary
        
        Args:
            array (thresh): Threshed array that will serve for drawing counters
            min_size (int, optional): minimal size of countour to be drawn
            max_size (int, optional): maxiaml size of countour to be drawn
            color (list or int, optional): color of countour
            thickness (int, optional): thickness of contour. -1 fills all

        Returns:
            canvas (RGB image): returns original image with contours
        """
        return Coat(self)

    #@copy_meta
    def replace(self, past_color: list, future_color: list, dtype=np.uint8) -> 'HigherCoating':
        """ replace particular color
        
        Args:
            past_color (list): color to be replace
            future_color (list): color to replace with
            dtype (TYPE, optional): output type
        
        Returns:
            self: Description
        """
        self[
            np.where(np.all(self == past_color, axis=len(self.shape) - 1))
        ] = future_color
        return dtype(self)

    #@copy_meta
    def to_color(self, color_transformation: str) -> 'HigherCoating':
        """ Change color space
        
        Args:
            color_transformation (str): predefined string
        
        Returns:
            HigherCoating: Description
        """
        return Coat(cv2.cvtColor(self, self.COLOR_SPACE[color_transformation]))

    def show(self) -> 'HigherCoating':
        """ Show array
        
        careful about unsigned8. Show float doesnt show the same image as uin8

        Returns:
            self: Description
        """
        cv2.imshow("array", self)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return self

    #@copy_meta
    def blur_median(self, value: int) -> 'HigherCoating':
        """ median blur
        
        Args:
            value (int): Description
        
        Returns:
            HigherCoating: Description
        """
        return Coat(cv2.medianBlur(self, value))
    #@copy_meta
    def filter_bilateral(self, sigmaColor: int, sigmaSpace: int, borderType: int) -> 'HigherCoating':
        """Bilateral filter
        
        Args:
            sigmaColor (int): Description
            sigmaSpace (int): Description
            borderType (int): Description
        
        Returns:
            HigherCoating: Description
        """
        # 9,75,75
        return Coat(cv2.bilateralFilter(self, sigmaColor, sigmaSpace, borderType))
    #@copy_meta
    def blur_gauss(self, kernel):
        return Coat(cv2.GaussianBlur(self, kernel, 0))
    #@copy_meta
    def blur_average(self, kernel):
        # kernel (5,5)
        return Coat(cv2.blur(self, kernel))

    def read(self, path, flag=1):
        if path.endswith("npy"):
            return self + Coat(np.load(path))
        else:
            return self + Coat(cv2.imread(path, flag))
    def save(self,filename):
        cv2.imwrite(filename,self)
        return self

    def extend(self, value: Union[float,int], top=2, bottom=2, left=2, right=2) -> 'HigherCoating':
        """ Extend array borders with contant value
        
        Args:
            value (Union[float, int]): border value
            top (int, optional): Description
            bottom (int, optional): Description
            left (int, optional): Description
            right (int, optional): Description
        
        Returns:
            self: Description
        """
        self = (
            Coat(
                cv2.copyMakeBorder(
                    self,
                    top=top,
                    bottom=bottom,
                    left=left,
                    right=right,
                    borderType=cv2.BORDER_CONSTANT,
                    value=value,
                )
            )
        )
        return self
    #@copy_meta
    def conv(self, kernel: Union[list,np.ndarray]) -> 'HigherCoating':
        """ 2D convolution
        
        Args:
            kernel (Union[list, np.ndarray]): Description
        
        Returns:
            HigherCoating: Description
        
        Raises:
            Exception: when no kernel provided
        """
        original_type = self.dtype
        if kernel is not None:
            if isinstance(kernel, str):
                kernel = self.KERNEL.get(kernel, None)
            elif isinstance(kernel, np.ndarray):
                pass
            else:
                kernel = None

        if kernel is not None:
            return (Coat(cv2.filter2D(self, -1, kernel))).astype(
                original_type
            )

        else:
            raise Exception("No kernel found")

    def _translate_corners(self, corner1, corner2):
        """ Auto translation from relative corner coordinates to absolute one
        
        Args:
            corner1 (tuple): Description
            corner2 (tuple): Description
        
        Returns:
            corners (tuples): Description
        """
        if any(
            [
                isinstance(corner1[0], float),
                isinstance(corner1[1], float),
                isinstance(corner2[0], float),
                isinstance(corner2[1], float),
            ]
        ):

            corner1 = int(corner1[0] * self.shape[0]), int(corner1[1] * self.shape[1])

            corner2 = int(corner2[0] * self.shape[0]), int(corner2[1] * self.shape[1])
            return corner1, corner2
        else:
            return corner1, corner2

    def box(self, corner1: tuple, corner2: tuple, color=[255, 255, 255], thickness=2, copy=True) -> 'HigherCoating':

        """ Draw box
        x1,y1 ------    corner1 = (x1,y1)
        |          |
        |          |
        |          |
        --------x2,y2   corner2 = (x2,y2)
        
        Args:
            corner1 (tuple): ints or floats, int - absolute value, float relative(scale) value 
            corner2 (tuple): ints or floats, int - absolute value, float relative(scale) value 
            color (list, optional): rgb value
            thickness (int, optional): Description
            copy (bool, optional): If you set True, you will get copy, if you set False, boxes are drawn on self
        
        Returns:
            HigherCoating: Description
        """
        corner1, corner2 = self._translate_corners(corner1, corner2)

        if copy:
            return cv2.rectangle(
                self.copy(), corner1[::-1], corner2[::-1], color, thickness
            )

        else:
            return cv2.rectangle(self, corner1[::-1], corner2[::-1], color, thickness)

    def label(
        self,
        text: str,
        corner: tuple,
        color=(255, 255, 255),
        flag=0,
        scale=1,
        ltype=2,
        font="FONT_HERSHEY_SIMPLEX",
        copy=True,
    ) -> 'HigherCoating':
        """Summary
        up=False -> text hangs on corner
        up=True -> text sits on corner
        Args:
            text (str): Description
            corner (tuple): Description
            color (tuple, optional): Description
            flag (int, optional): Description
            scale (int, optional): Description
            ltype (int, optional): Description
            font (str, optional): Description
            copy (bool, optional): Description
        """
        font = self.FONT[font]
        bottomLeftCornerOfText = corner
        fontScale = scale
        fontColor = color
        lineType = ltype

        text_width, text_height = cv2.getTextSize(text, font, fontScale, lineType)[0]

        if flag:
            bottomLeftCornerOfText = tuple([corner[0] + text_height, corner[1]])

        if copy:
            return cv2.putText(
                self.copy(),
                text,
                bottomLeftCornerOfText[::-1],
                font,
                fontScale,
                fontColor,
                lineType,
            )
        else:
            return cv2.putText(
                self,
                text,
                bottomLeftCornerOfText[::-1],
                font,
                fontScale,
                fontColor,
                lineType,
            )

    #@copy_meta
    def labelbox(
        self,
        text: str,
        corner1: tuple,
        corner2: tuple,
        up=False,
        copy=True,
        bcolor=[255, 255, 255],
        tcolor=[0, 0, 0],
        text_size=15,
        align="center",
    ) -> 'HigherCoating':

        if corner1 is None:
            corner1 = (0, 0)
        if corner2 is None:
            corner2 = (self.shape[0], self.shape[1])

        corner1, corner2 = self._translate_corners(corner1, corner2)

        if align == "left":
            text = f"{text:<{text_size}}"
        elif align == "right":
            text = f"{text:>{text_size}}"
        elif align == "center":
            text = f"{text:^{text_size}}"
        else:
            raise Exception("Wrong option")
        label = Coat(create_label(text, bcolor=bcolor, tcolor=tcolor))
        image = self.box(corner1, corner2, color=bcolor, thickness=1, copy=copy)
        width = corner2[1] - corner1[1]
        ratio = width / label.shape[1]
        label = label.osize((int(ratio * label.shape[0]), corner2[1] - corner1[1]))
        if len(label.shape) == 2 and len(image.shape) == 3:
            label = label.astype(np.uint8).to_color("GRAY2BGR")
        if len(label.shape) == 3 and len(image.shape) == 2:
            print(label.shape)
            label = label.astype(np.uint8).to_color("BGR2GRAY")

        image_shape = image.shape

        x = max(
            abs(corner1[0]) if corner1[0] < 0 else 0,
            corner2[0] - image_shape[0] if corner2[0] - image_shape[0] > 0 else 0,
        )
        y = max(
            abs(corner1[1]) if corner1[1] < 0 else 0,
            corner2[1] - image_shape[1] if corner2[1] - image_shape[1] > 0 else 0,
        )
        max_overlay = max(x, y) + max(label.shape) if up else max(x, y)

        if max_overlay:
            image = image.extend(
                [0, 0, 0],
                top=max_overlay,
                bottom=max_overlay,
                right=max_overlay,
                left=max_overlay,
            )

        if not up:

            x_int = max_overlay + corner1[0], max_overlay + label.shape[0] + corner1[0]
            y_int = max_overlay + corner1[1], max_overlay + corner1[1] + width
            image[x_int[0] : x_int[1], y_int[0] : y_int[1]] = label
        if up:
            x_int = max_overlay + corner1[0] - label.shape[0], max_overlay + corner1[0]
            y_int = max_overlay + corner1[1], max_overlay + corner1[1] + width

            image[x_int[0] : x_int[1], y_int[0] : y_int[1]] = label

        if max_overlay:
            return image[max_overlay:-max_overlay, max_overlay:-max_overlay]
        else:
            return image

    def join(self,other: Union[np.ndarray,'HigherCoating'],direction=0) -> 'HigherCoating':
        """ Montage of two images.

        Connect two images next to each other
        
        Args:
            other (Union[np.ndarray, 'HigherCoating']): Description
            direction (int, optional): to 0 - right or 1 - down
        
        Returns:
            HigherCoating: Description
        """
        grid_direction = {1:(2,1),0:(1,2)}
        if isinstance(other,np.ndarray):
            if self.dominant:
                template = self.copy()
            elif not self.dominant and other.dominant:
                template = other.copy()
            else:
                template = self.copy()

            return Montage([self,other]).template(template).grid(*grid_direction[direction])
        else:
            return self
    #@copy_meta
    def filter(self,lower: list,upper: list,passband = False) -> 'HigherCoating':
        """ Filter values
        
        Args:
            lower (list): color
            upper (list): color
            passband (bool, optional): passband -> filter colored interval or everything except colored interval
        
        Returns:
            HigherCoating: Description
        """
        lower = np.array(lower)
        upper = np.array(upper)
        mask = cv2.inRange(self,lower,upper)

        if not passband:
            return (Coat(cv2.bitwise_and(self,self, mask= mask)))
        else:
            return (Coat(cv2.bitwise_and(self,self, mask= np.logical_not(mask).astype(np.uint8))))

    #@copy_meta
    def filterHsv(self,lower: list,upper: list,passband = False) -> 'HigherCoating':

        """Filter values ( It is expected you have BGR)
        BGR->HSV transformation
    
        Args:
            lower (list): HSV color
            upper (list): HSV color
            passband (bool, optional): passband -> filter colored interval or everything except colored interval
        
        Returns:
            HigherCoating: Description
        
        """
        lower = np.array(lower)
        upper = np.array(upper)
        hsv = cv2.cvtColor(self,cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv,lower,upper)
        if not passband:
            return (Coat(cv2.bitwise_and(hsv,hsv, mask= mask))).to_color('HSV2BGR')
        else:
            return (Coat(cv2.bitwise_and(hsv,hsv, mask= np.logical_not(mask).astype(np.uint8)))).to_color('HSV2BGR')





class Montage:
    def __init__(self, list_of_images: list):
        """ Constructor        
        Args:
            list_of_images (list): List of image of any size, dimension(2 or 3) or data type
        """
        self.images = list_of_images
        self.tpl = None

    def _filling(self, vertical: int, horizontal: int, fill_direction=0) -> GeneratorType:
        """ Create 2D range generator based on fillinf direction (left to right or top to down)
        
        Args:
            vertical (int): Description
            horizontal (int): Description
            fill_direction (int, optional): Description
        
        Yields:
            GeneratorType: Description
        """
        if fill_direction:
            for v in range(vertical):
                for h in range(horizontal):
                    yield v, h
        else:
            for h in range(horizontal):
                for v in range(vertical):
                    yield v, h

    def grid(self, vertical: int, horizontal: int, direction=0) -> 'HigherCoating':
        """ Create montage
        
        Args:
            vertical (int): number of rows
            horizontal (int): number of columns
            direction (int, optional): 0 right to left, 1 top to down
        
        Returns:
            HigherCoating: Description
        """
        if self.tpl is None:
            self.tpl = self.images[0]
        if self.tpl is not None:
            shape = self.tpl.shape
            template_shape = (
                [shape[0] * vertical, shape[1] * horizontal]
                if len(shape) == 2
                else [shape[0] * vertical, shape[1] * horizontal, 3]
            )
            montage = np.zeros(shape=template_shape).astype(self.tpl.dtype)

        counter, number_of_images = 0, len(self.images)
        for v, h in self._filling(vertical, horizontal, fill_direction=direction):
            montage[
                v * shape[0] : (v + 1) * shape[0], h * shape[1] : (h + 1) * shape[1]
            ] = HigherCoating(shape=self.tpl.shape).astype(
                self.tpl.dtype
            ).host() * 0 + Coat(
                self.images[counter]
            )
            counter += 1
            if counter >= number_of_images:
                break
        return Coat(montage)

    def template(self, template: Union[np.ndarray,'HigherCoating']) -> 'HigherCoating':
        """ Set Montage cell common standard
        
        Args:
            template (Union[np.ndarray, 'HigherCoating']): If you add array of shape 100,100,3, you will have RGB montage
        
        Returns:
            self: Description
        """
        self.tpl = template
        return self
