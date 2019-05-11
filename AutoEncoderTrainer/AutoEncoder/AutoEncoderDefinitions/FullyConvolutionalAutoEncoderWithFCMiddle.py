
from keras.layers import Input, Dense, Conv2D, Flatten, Dropout, MaxPooling2D, Reshape, UpSampling2D
from keras.models import Model

from AutoEncoderTrainer.AutoEncoder.AutoEncoder import AutoEncoder


class FullyConvolutionalAutoEncoderWithFCMiddle(AutoEncoder):
    def get_num_decoding_layers_to_rip_out(self):
        return 3

    def __init__(self, model_hyper_parameters):
        AutoEncoder.__init__(self, model_hyper_parameters)

    def create_autoencoder(self, input_image_vector):
        input_image = Input(shape=(self.image_width_after_rescale,
                                   self.image_height_after_rescale,
                                   self.image_depth_after_rescale))
        encoded = Conv2D(256, kernel_size=(4, 4), activation='linear', padding='same', kernel_initializer=self.get_initializer())(input_image)
        encoded = MaxPooling2D(pool_size=(2, 2), padding='same')(encoded)
        encoded = Conv2D(128, kernel_size=(4, 4), activation='linear',  padding='same', kernel_initializer=self.get_initializer())(encoded)
        encoded = MaxPooling2D(pool_size=(2, 2), padding='same')(encoded)
        encoded = Conv2D(64, kernel_size=(4, 4), activation='linear', padding='same', kernel_initializer=self.get_initializer())(encoded)
        encoded = MaxPooling2D(pool_size=(2, 2), padding='same')(encoded)
        encoded = Conv2D(32, kernel_size=(4, 4), activation='linear', padding='same', kernel_initializer=self.get_initializer())(encoded)
        encoded = MaxPooling2D(pool_size=(2, 2), padding='same')(encoded)
        encoded = Flatten()(encoded)
        encoded = Dense(14 * 14 * 32, activation='linear', kernel_initializer=self.get_initializer())(encoded)
        encoded = Dense(1024, activation='linear', kernel_initializer=self.get_initializer())(encoded)
        decoded = Dense(14 * 14 * 32, activation='linear', kernel_initializer=self.get_initializer())(encoded)
        decoded = Reshape((14, 14, 32))(decoded)
        decoded = Conv2D(64, (4, 4), activation='linear', padding='same', kernel_initializer=self.get_initializer())(UpSampling2D(size=(2, 2))(decoded))
        decoded = Conv2D(128, (4, 4), activation='linear', padding='same', kernel_initializer=self.get_initializer())(UpSampling2D(size=(2, 2))(decoded))
        decoded = Conv2D(256, (4, 4), activation='linear', padding='same', kernel_initializer=self.get_initializer())(UpSampling2D(size=(2, 2))(decoded))
        decoded = Conv2D(3, 1, activation='linear', padding='same', kernel_initializer=self.get_initializer())(UpSampling2D(size=(2, 2))(decoded))
        decoded = Flatten()(decoded)
        auto_encoder = Model(input_image, decoded)
        self.compile_autoencoder(auto_encoder)
        print(auto_encoder.summary())
        return auto_encoder