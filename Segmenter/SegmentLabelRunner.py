import copy
import sys
import cv2
from keras.engine.saving import load_model
from keras.optimizers import Adam
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import tensorflow as tf
from Segmenter.zf_unet_224_model import ZF_UNET_224, dice_coef_loss, dice_coef






if __name__ == "__main__":
    image = cv2.imread("/home/hujohnso/Documents/Research2018/FrameExtractor/Animations/squareCircleStar.0007.png")
    image = cv2.resize(image, (224, 224))

    for object_number in set_of_objects:
        print_image_by_value(object_number, new_image)




