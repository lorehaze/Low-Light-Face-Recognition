from re import I, X
from PIL import Image
from math import sqrt
import io
from skimage import exposure
from skimage import io

# folder -> photo directory
# photo_toCheck -> photo name to check for brightness
# folder = "photos/"
# photo_name = "Yellow.png"
# photo_toCheck = str(folder + photo_name)

# set default standard black brightness
# reference https://a.atmos.washington.edu/~ovens/javascript/colorpicker.html
# insert default RGB values for a standard darked photo
std_R = 64
std_G = 64
std_B = 64
stdRGBMean = (
    std_B + std_G + std_R
) / 3  # calculate standard mean value from parameters

# path to get photo from
folder = "photos/"
photo_name = "images.jpeg"
photo_toCheck = str(folder + photo_name)

# path to save post-elaboration output
output_folder = "equalized/"
output_path = str(output_folder)


def brightnessCalculator(path_to_photo_toCheck):
    imag = Image.open(path_to_photo_toCheck)
    # Convert the image te RGB if it is a .gif for example
    imag = imag.convert("RGB")
    # coordinates of the pixel
    X, Y = 0, 0
    # Get RGB
    pixelRGB = imag.getpixel((X, Y))
    R, G, B = pixelRGB
    brightness = float(
        sum([R, G, B]) / 3
    )  # 0 is dark (black) and 255 is bright (white)
    return brightness


def photoBrightnessEvaluate(path_to_photo_toCheck):
    tmpPhotoBrightness = brightnessCalculator(
        path_to_photo_toCheck
    )  # calculate photo brightness
    if tmpPhotoBrightness < stdRGBMean:  # compare with standard medium value
        return True


def photoProcessor(boolean_value):
    if processingFlag == True:
        print("Equalization started.")
        img = io.imread(photo_toCheck)
        # Contrast stretching
        # p2, p98 = np.percentile(img, (2, 98))
        # img_rescale = exposure.rescale_intensity(img, in_range=(p2, p98))
        # Adaptive Equalization
        img_adapteq = exposure.equalize_adapthist(img, clip_limit=0.03)
        io.imsave(
            output_folder + photo_name,
            img_adapteq,
        )


# Example
processingFlag = photoBrightnessEvaluate(photo_toCheck)
photoProcessor(processingFlag)
