from AutoEncoder.AutoEncoder import AutoEncoder
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Flatten, BatchNormalization, LeakyReLU, Add, \
    Dropout
from keras import regularizers
from keras.legacy import layers
from keras.models import Sequential, Model
import matplotlib.pyplot as plt
import time
import numpy
from skimage import data, img_as_float, img_as_int
from skimage.transform import rescale
from keras.callbacks import TensorBoard


class MNISTExampleAutoEncoder(AutoEncoder):
    def __init__(self, model_hyper_parameters):
        AutoEncoder.__init__(self, model_hyper_parameters)

    def create_autoencoder(self, input_image_vector):
        flattened_vector_size = self.image_width_after_rescale * self.image_height_after_rescale * self.image_depth_after_rescale
        input_image = Input(shape=(self.image_width_after_rescale,
                                   self.image_height_after_rescale,
                                   self.image_depth_after_rescale))
        encoded = Conv2D(32, kernel_size=(3, 3), activation='relu')(input_image)
        encoded = Conv2D(64, (3, 3), activation='relu')(encoded)
        encoded = Dropout(0.25)(encoded)
        encoded = Flatten()(encoded)
        decoded = Dense(128, activation='relu')(encoded)
        decoded = Dropout(0.5)(decoded)
        decoded = Dense(flattened_vector_size, activation='softmax')(decoded)
        auto_encoder = Model(input_image, decoded)
        auto_encoder.compile(optimizer='adam', loss='mean_squared_error', metrics=['binary_crossentropy'])
        print(auto_encoder.summary())
        return auto_encoder