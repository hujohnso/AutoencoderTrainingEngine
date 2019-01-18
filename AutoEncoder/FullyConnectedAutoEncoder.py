from AutoEncoder.AutoEncoder import AutoEncoder
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Flatten, BatchNormalization, LeakyReLU, Add
from keras import regularizers
from keras.legacy import layers
from keras.models import Sequential, Model
import matplotlib.pyplot as plt
import time
import numpy
from skimage import data, img_as_float
from skimage.transform import rescale
from keras.callbacks import TensorBoard
import random as rand


class FullyConnectedAutoEncoder(AutoEncoder):
    def __init__(self):
        super.__init__()

    def create_auto_encoder(self, input_image_vector):
        flattened_vector_size = self.image_width_after_rescale * self.image_height_after_rescale * self.image_depth_after_rescale
        input_image = Input(shape=(self.image_width_after_rescale,
                                   self.image_height_after_rescale,
                                   self.image_depth_after_rescale))
        encoded = Flatten()(input_image)
        encoded = Dense(flattened_vector_size,
                        activation='linear', kernel_initializer=self.get_initializer(),
                        activity_regularizer=self.get_regularizer())(encoded)
        encoded = Dense(int (flattened_vector_size * .1),
                        activation= 'linear',
                        kernel_initializer= self.get_initializer())(encoded)
        decoded = Dense(flattened_vector_size,
                        activation='linear', kernel_initializer=self.get_initializer())(encoded)
        auto_encoder = Model(input_image, decoded)
        auto_encoder.compile(optimizer='adam', loss='mean_squared_error', metrics=['binary_crossentropy'])
        print(auto_encoder.summary())
        return auto_encoder

    def visualize(self, trained_model):
        image = img_as_float(data.load(self.hyper_params.filePath + self.hyper_params.input_image_name,
                                       as_gray=self.hyper_params.as_gray))
        fig, axes = plt.subplots(1, 2)
        ax = axes.flatten()
        ax[0].set_title("Original Image", fontsize=12)
        ax[0].imshow(image)
        ax[0].set_axis_off()
        ax[1].set_title("Image after auto encoder", fontsize=12)
        image_after_encoder = self.reformat_auto_encoder_format(trained_model.predict(self.prepare_single_image(image)))
        ax[1].imshow(image_after_encoder)
        ax[1].set_axis_off()
        fig.tight_layout()
        plt.show()

