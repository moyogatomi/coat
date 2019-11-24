import numpy as np
import cv2
import urllib
import validators


def create_label(text, bcolor=[255, 255, 255], tcolor=[0, 0, 0]):
    y, x = cv2.getTextSize(text, cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, 1)[0]
    text_image = np.ones(shape=[int(x * 1.45), int(y * 1.02), 3]) * np.array(bcolor)
    print(y, x)

    return cv2.putText(
        text_image,
        text,
        (0, x - int(x * 0.02)),
        cv2.FONT_HERSHEY_COMPLEX_SMALL,
        1,
        tcolor,
        1,
    )

def urlvalidation(url):
    valid = validators.url(url)
    if valid:
        return True
    else:
        return False

def url_to_image(url):
    resp = urllib.request.urlopen(url)
    image = np.asarray(bytearray(resp.read()), dtype="uint8")
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    return image