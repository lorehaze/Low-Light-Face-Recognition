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

# path to get photo from
folder = "photos/"
photo_name = "grey_stretched.jpg"
photo_toCheck = str(folder + photo_name)

# path to save post-elaboration output
output_folder = "equalized/"
output_path = str(output_folder)

processingFlag = photoBrightnessEvaluate(photo_toCheck)


if processingFlag == True:
    img = io.imread(photo_toCheck)
    # Contrast stretching
    p2, p98 = np.percentile(img, (2, 98))
    img_rescale = exposure.rescale_intensity(img, in_range=(p2, p98))
    io.imsave(
        output_folder + "contrast_stretched.jpg",
        img_rescale,
    )
