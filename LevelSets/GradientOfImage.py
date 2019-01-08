import numpy as np
import scipy.ndimage
import scipy.signal
import matplotlib.pyplot as plt
from skimage import color, io


def grad(x):
    return np.array(np.gradient(x))


def norm(x, axis=0):
    return np.sqrt(np.sum(np.square(x), axis=axis))


def stopping_fun(x):
    return 1. / (1. + norm(grad(x))**2)


img = io.imread('Cute Baby Holland Lop Bunnies Playing Inside the House.mp4frame235.jpg')
img = color.rgb2gray(img)
img = img - np.mean(img)

# Smooth the image to reduce noise and separation between noise and edge becomes clear
img_smooth = scipy.ndimage.filters.gaussian_filter(img, 1)
plt.imshow(img_smooth)
plt.show()
F = stopping_fun(img_smooth)