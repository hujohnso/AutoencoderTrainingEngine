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


class ConvAutoEncoder(AutoEncoder):
    def __init__(self, model_hyper_parameters):
        AutoEncoder.__init__(self, model_hyper_parameters)

    def create_auto_encoder(self, input_image_vector):
        input_image = Input(shape=(input_image_vector.shape[1],
                                   input_image_vector.shape[2],
                                   input_image_vector.shape[3]))
        encoded = Flatten()(input_image)
        encoded = Dense(input_image_vector.shape[1] * input_image_vector.shape[2] * input_image_vector.shape[3],
                        activation='linear', kernel_initializer=self.get_initializer(),
                        activity_regularizer=self.get_regularizer())(encoded)
        encoded = Dense(int (input_image_vector.shape[1] * input_image_vector.shape[2] * input_image_vector.shape[3] * .1),
                        activation= 'linear',
                        kernel_initializer= self.get_initializer())(encoded)
        decoded = Dense(input_image_vector.shape[1] * input_image_vector.shape[2] * input_image_vector.shape[3],
                        activation='linear', kernel_initializer=self.get_initializer())(encoded)
        autoEncoder = Model(input_image, decoded)
        autoEncoder.compile(optimizer='adam', loss='mean_squared_error', metrics=['binary_crossentropy'])
        print(autoEncoder.summary())
        return autoEncoder

    def residual_block(self, y, nb_channels, _strides=(1, 1), _project_shortcut=False):
        shortcut = y

        # down-sampling is performed with a stride of 2
        y = Conv2D(nb_channels,
                   kernel_size=(3, 3),
                   strides=_strides,
                   padding='same',
                   kernel_initializer=self.get_initializer(),
                   bias_initializer= self.get_bias_initializer())(y)
        # y = MaxPooling2D((2, 2), padding='same')(y)
        y = BatchNormalization()(y)
        y = LeakyReLU()(y)

        y = Conv2D(nb_channels,
                   kernel_size=(3, 3),
                   strides=(1, 1),
                   padding='same',
                   kernel_initializer=self.get_initializer(),
                   bias_initializer=self.get_bias_initializer())(y)
        # y = MaxPooling2D((2, 2), padding='same')(y)
        y = BatchNormalization()(y)
        if _project_shortcut or _strides != (1, 1):
           # when the dimensions increase projection shortcut is used to match dimensions (done by 1×1 convolutions)
           # when the shortcuts go across feature maps of two sizes, they are performed with a stride of 2
            shortcut = Conv2D(nb_channels,
                              kernel_size=(1, 1),
                              strides=_strides,
                              padding='same',
                              kernel_initializer=self.get_initializer(),
                              bias_initializer=self.get_bias_initializer())(shortcut)
           # y = MaxPooling2D((2, 2), padding='same')(y)
            shortcut = BatchNormalization()(shortcut)
        y = Add()([shortcut, y])
        y = LeakyReLU()(y)

        return y
 ##Use tensor board, save model, and read unet paper
    # selbeleidy@mymail.mines.edu

    def visualize(self, trained_model):
        image = img_as_float(data.load(self.hyper_params.file_path + self.hyper_params.test_image_name,
                                       as_gray=self.hyper_params.as_gray))
        fig, ax = plt.subplots(1, 3)
        ax[0].set_title("Original Image", fontsize=12)
        ax[0].imshow(image)
        ax[0].set_axis_off()

        ax[1].set_title("image fed in", fontsize=12)
        ax[1].imshow(self.reformat_auto_encoder_format(self.prepare_single_image(image)))
        ax[1].set_axis_off()

        ax[2].set_title("Image after auto encoder", fontsize=12)
        image_after_encoder = self.reformat_auto_encoder_format(
            trained_model.predict(self.prepare_single_image(image).reshape(1,
                                                                           self.image_width_after_rescale,
                                                                           self.image_height_after_rescale,
                                                                           self.image_depth_after_rescale)))
        ax[2].imshow(image_after_encoder)
        ax[2].set_axis_off()

        fig.tight_layout()
        plt.show()
