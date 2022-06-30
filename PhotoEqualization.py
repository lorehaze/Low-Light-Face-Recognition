from PhotoPreProcesing import (
    photoBrightnessEvaluate,
)  # import custom brightness evaluator
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import skimage.io
import io
import keras.preprocessing.image
from skimage import data, img_as_float
from skimage import exposure
import cv2
from skimage import io

folder = "photos/"
photo_name = "Dark1.jpg"
photo_toCheck = str(folder + photo_name)

processingFlag = photoBrightnessEvaluate(photo_toCheck)


if processingFlag == True:
    img = io.imread(
        "/Users/lorenzo/Documents/GitHub/Low-Light-Face-Recognition/photos/Dark1.jpg"
    )
    # Contrast stretching
    p2, p98 = np.percentile(img, (2, 98))
    img_rescale = exposure.rescale_intensity(img, in_range=(p2, p98))
    io.imsave(
        "/Users/lorenzo/Documents/GitHub/Low-Light-Face-Recognition/equalized/contrast_stretched.jpg",
        img_rescale,
    )
