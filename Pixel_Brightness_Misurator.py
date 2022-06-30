from concurrent.futures import process
from re import I, X
from PIL import Image
from math import sqrt
from matplotlib import image

#reference https://a.atmos.washington.edu/~ovens/javascript/colorpicker.html
standard_black_brightness_value=[89,89,89]
count = 0
processing_flag = False

for x in standard_black_brightness_value:
    sum = +x
    count += 1
    standard_value = sum / count


standard_value = standard_black_brightness_value

def brightness_calculator(path_to_photo_toCheck):
    imag = Image.open(path_to_photo_toCheck)
    #Convert the image te RGB if it is a .gif for example
    imag = imag.convert ('RGB')
    #coordinates of the pixel
    X,Y = 0,0
    #Get RGB
    pixelRGB = imag.getpixel((X,Y))
    R,G,B = pixelRGB 
    brightness = sum([R, G, B])  # 0 is dark (black) and 255 is bright (white)
    return brightness

def comparison (to_compare):
    if to_compare < standard_value:
    #    processing_flag = True
        return True