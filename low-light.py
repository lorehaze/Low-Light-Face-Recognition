#import librerie
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import skimage.io
import io
#from scipy.misc import imsave
#from keras.preprocessing.image import save_img
import keras.preprocessing.image
from skimage import data, img_as_float
from skimage import exposure
#import imread
import cv2
from skimage import io
#from google.colab.patches import cv2_imshow

# edit default runtime configuration for matplotlib, font size set to 8 for bettere visibility
matplotlib.rcParams['font.size'] = 8

# function that plots image and histogram
def plot_img_and_hist(image, axes, bins=256):
    """Plot an image along with its histogram and cumulative histogram.

    """
    image = img_as_float(image)
    ax_img, ax_hist = axes
    ax_cdf = ax_hist.twinx()

    # Display image
    ax_img.imshow(image, cmap=plt.cm.gray)
    ax_img.set_axis_off()
    ax_img.set_adjustable('box')

    # Display histogram
    ax_hist.hist(image.ravel(), bins=bins, histtype='step', color='black')
    ax_hist.ticklabel_format(axis='y', style='scientific', scilimits=(0, 0))
    ax_hist.set_xlabel('Pixel intensity')
    ax_hist.set_xlim(0, 1)
    ax_hist.set_yticks([])

    # Display cumulative distribution
    img_cdf, bins = exposure.cumulative_distribution(image, bins)
    ax_cdf.plot(bins, img_cdf, 'r')
    ax_cdf.set_yticks([])

    return ax_img, ax_hist, ax_cdf


# Load an example image
#img = data.imread('/home/fakrul/Downloads/kkkk.jpg')/home/fakrul/Documents/thesis /images
img = io.imread('/Users/lorenzo/Desktop/Low-Light/images/darked_face.jpg')
img_o = io.imread(
    '/Users/lorenzo/Desktop/Low-Light/images/darked_face.jpg')
img_p = io.imread(
    '/Users/lorenzo/Desktop/Low-Light/images/darked_face.jpg')
img_q = io.imread(
    '/Users/lorenzo/Desktop/Low-Light/images/darked_face.jpg')
io.imsave('/Users/lorenzo/Desktop/Low-Light/face/darked_face.jpg', img,
          check_contrast=False)  # check_contrast=False disables warning

# Contrast stretching
p2, p98 = np.percentile(img, (2, 98))
img_rescale = exposure.rescale_intensity(img, in_range=(p2, p98))
io.imsave('/Users/lorenzo/Desktop/Low-Light/face/contrast_stretching/contrast_stretched.jpg', img_rescale)

# Equalization
img_eq = exposure.equalize_hist(img)
io.imsave('/Users/lorenzo/Desktop/Low-Light/face/histogram_equalization/histogram_equalized.jpg', img_eq)

# Adaptive Equalization
img_adapteq = exposure.equalize_adapthist(img, clip_limit=0.03)
io.imsave('/Users/lorenzo/Desktop/Low-Light/face/adaptive_equalization/adaptive_equalized.jpg', img_adapteq)

# Display results
fig = plt.figure(figsize=(8, 5))
axes = np.zeros((2, 4), dtype=object)
axes[0, 0] = fig.add_subplot(2, 4, 1)
for i in range(1, 4):
    axes[0, i] = fig.add_subplot(
        2, 4, 1+i, sharex=axes[0, 0], sharey=axes[0, 0])
for i in range(0, 4):
    axes[1, i] = fig.add_subplot(2, 4, 5+i)

ax_img, ax_hist, ax_cdf = plot_img_and_hist(img, axes[:, 0])
ax_img.set_title('Low contrast image')

y_min, y_max = ax_hist.get_ylim()
ax_hist.set_ylabel('Number of pixels')
ax_hist.set_yticks(np.linspace(0, y_max, 5))

ax_img, ax_hist, ax_cdf = plot_img_and_hist(img_rescale, axes[:, 1])
ax_img.set_title('Contrast stretching')

ax_img, ax_hist, ax_cdf = plot_img_and_hist(img_eq, axes[:, 2])
ax_img.set_title('Histogram equalization')

ax_img, ax_hist, ax_cdf = plot_img_and_hist(img_adapteq, axes[:, 3])
ax_img.set_title('Adaptive equalization')

ax_cdf.set_ylabel('Fraction of total intensity')
ax_cdf.set_yticks(np.linspace(0, 1, 10))

# prevent overlap of y-axis labels
fig.tight_layout()
plt.show()
# img.dtype
# img.min(),img.max()

# loading face and eye cascades -
face_cascade = cv2.CascadeClassifier(
    '/Users/lorenzo/Desktop/Low-Light/cascades/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(
    '/Users/lorenzo/Desktop/Low-Light/cascades/haarcascade_eye.xml')

# Try to detect contrast stretched
img = cv2.imread(
    '/Users/lorenzo/Desktop/Low-Light/face/contrast_stretching/contrast_stretched.jpg')

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

faces = face_cascade.detectMultiScale(gray, 1.1, 5)
for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
    cv2.rectangle(img_o, (x, y), (x+w, y+h), (255, 0, 0), 2)
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = img[y:y+h, x:x+w]

    cv2.imwrite(
        '/Users/lorenzo/Desktop/Low-Light/face/contrast_stretching/cropped.jpg', roi_color)

    # cv2.imwrite('gyt.jpg',roi_color)
    #eyes = eye_cascade.detectMultiScale(roi_gray)
    cv2.imshow('Contrast Stretching', img_o)
    io.imsave(
        '/Users/lorenzo/Desktop/Low-Light/face/contrast_stretching/detect.jpg', img_o)
    # for (ex,ey,ew,eh) in eyes:
    #   cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
    #    cv2_imshow(roi_color)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    img = cv2.imread(
        '/Users/lorenzo/Desktop/Low-Light/face/histogram_equalization/histogram_equalized.jpg')

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

faces = face_cascade.detectMultiScale(gray, 1.1, 5)
for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
    cv2.rectangle(img_p, (x, y), (x+w, y+h), (255, 0, 0), 2)
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = img[y:y+h, x:x+w]

    cv2.imwrite(
        '/Users/lorenzo/Desktop/Low-Light/face/histogram_equalization/crop.jpg', roi_color)

    # cv2.imwrite('gyt.jpg',roi_color)
    #eyes = eye_cascade.detectMultiScale(roi_gray)
    cv2.imshow('Histogram Equalization', img_p)
    io.imsave(
        '/Users/lorenzo/Desktop/Low-Light/face/histogram_equalization/detect.jpg', img_p)

    # Eye detection
    # for (ex,ey,ew,eh) in eyes:
    #    cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
    #    cv2_imshow(roi_color)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

img = cv2.imread(
    '/Users/lorenzo/Desktop/Low-Light/face/adaptive_equalization/adaptive_equalized.jpg')

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

faces = face_cascade.detectMultiScale(gray, 1.1, 5)
for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
    cv2.rectangle(img_q, (x, y), (x+w, y+h), (255, 0, 0), 2)
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = img[y:y+h, x:x+w]

    cv2.imwrite(
        '/Users/lorenzo/Desktop/Low-Light/face/adaptive_equalization/crop.jpg', roi_color)

    # cv2.imwrite('gyt.jpg',roi_color)
    #eyes = eye_cascade.detectMultiScale(roi_gray)
    cv2.imshow('Adaptive Equalization', img_q)
    io.imsave(
        '/Users/lorenzo/Desktop/Low-Light/face/adaptive_equalization/detect.jpg', img_q)
    # for (ex,ey,ew,eh) in eyes:
    # cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
    # cv2.imshow('img',roi_color)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
