from AutoEncoder.AutoEncoder import AutoEncoder
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Flatten, BatchNormalization, LeakyReLU, Add
from keras import regularizers
from keras.legacy import layers
from keras.models import Sequential, Model
import matplotlib.pyplot as plt
import time
import numpy
from skimage import data, img_as_float, img_as_int
from skimage.transform import rescale
from keras.callbacks import TensorBoard

#
class ConvAutoEncoder(AutoEncoder):
    def __init__(self, model_hyper_parameters):
        AutoEncoder.__init__(self, model_hyper_parameters)

    def create_autoencoder(self, input_image_vector):
        flattened_vector_size = self.image_width_after_rescale * self.image_height_after_rescale * self.image_depth_after_rescale
        input_image = Input(shape=(self.image_width_after_rescale,
                                   self.image_height_after_rescale,
                                   self.image_depth_after_rescale))
        encoded = Conv2D(16, (3, 3), activation='relu', padding='same')(input_image)
        encoded = Flatten()(encoded)
        decoded = Dense(flattened_vector_size,
                        activation='relu', kernel_initializer=self.get_initializer(),
                        activity_regularizer=self.get_regularizer())(encoded)
        auto_encoder = Model(input_image, decoded)
        auto_encoder.compile(optimizer='adam', loss='mean_squared_error', metrics=['binary_crossentropy'])
        print(auto_encoder.summary())
        return auto_encoder
