
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from keras.models import Model

from AutoEncoderTrainer.AutoEncoder.AutoEncoder import AutoEncoder


class AlexNetConvolutionalAutoEncoder(AutoEncoder):
    def __init__(self, model_hyper_parameters):
        AutoEncoder.__init__(self, model_hyper_parameters)

    def create_autoencoder(self, input_image_vector):
        flattened_vector_size = self.image_width_after_rescale * self.image_height_after_rescale * self.image_depth_after_rescale
        input_image = Input(shape=(self.image_width_after_rescale,
                                   self.image_height_after_rescale,
                                   self.image_depth_after_rescale))
        encoded = Conv2D(filters=96, kernel_size=(3, 3), strides=(2, 2), padding='same', activation= 'relu')(input_image)
        encoded = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(encoded)
        encoded = Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(encoded)
        encoded = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(encoded)
        encoded = Conv2D(filters=384, kernel_size=(3, 3), strides=(1, 1), padding='same', activation= 'relu')(encoded)
        encoded = Conv2D(filters=384, kernel_size=(3, 3), strides=(1, 1), padding='same', activation= 'relu')(encoded)
        encoded = Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same', activation= 'relu')(encoded)
        encoded = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(encoded)
        encoded = Flatten()(encoded)
        decoded = Dense(4096, activation='relu')(encoded)
        decoded = Dropout(0.4)(decoded)
        decoded = Dense(4096, activation='relu')(decoded)
        decoded = Dropout(0.4)(decoded)
        decoded = Dense(flattened_vector_size, activation='relu')(decoded)
        auto_encoder = Model(input_image, decoded)
        self.compile_autoencoder(auto_encoder)
        print(auto_encoder.summary())
        return auto_encoder


