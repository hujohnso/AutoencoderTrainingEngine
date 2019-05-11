
from keras.layers import Input, Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from keras.models import Model

from AutoEncoderTrainer.AutoEncoder.AutoEncoder import AutoEncoder


class SmallConvolutionalAutoEncoder(AutoEncoder):
    def get_num_decoding_layers_to_rip_out(self):
        return 3

    def __init__(self, model_hyper_parameters):
        AutoEncoder.__init__(self, model_hyper_parameters)

    def create_autoencoder(self, input_image_vector):
        flattened_vector_size = self.image_width_after_rescale * self.image_height_after_rescale * self.image_depth_after_rescale
        input_image = Input(shape=(self.image_width_after_rescale,
                                   self.image_height_after_rescale,
                                   self.image_depth_after_rescale))
        encoded = Conv2D(256, kernel_size=(3, 3), activation='linear', kernel_initializer=self.get_initializer())(input_image)
        encoded = MaxPooling2D(pool_size=(3, 3))(encoded)
        # encoded = Conv2D(256, (5, 5), activation='linear', kernel_initializer=self.get_initializer())(encoded)
        # encoded = MaxPooling2D(pool_size=(3, 3))(encoded)
        # encoded = Conv2D(256, (5, 5), activation='linear', kernel_initializer=self.get_initializer())(encoded)
        # encoded = MaxPooling2D(pool_size=(3, 3))(encoded)
        encoded = Conv2D(128, (3, 3), activation='linear', kernel_initializer=self.get_initializer())(encoded)
        encoded = MaxPooling2D(pool_size=(3, 3))(encoded)
        encoded = Conv2D(64, (3, 3), activation='linear', kernel_initializer=self.get_initializer())(encoded)
        encoded = MaxPooling2D(pool_size=(3, 3))(encoded)
        encoded = Flatten()(encoded)
        decoded = Dense(64, activation='linear', kernel_initializer=self.get_initializer())(encoded)
        # decoded = Dense(128, activation='linear', kernel_initializer=self.get_initializer())(decoded)
        decoded = Dense(flattened_vector_size, activation='linear', kernel_initializer=self.get_initializer())(decoded)
        auto_encoder = Model(input_image, decoded)
        self.compile_autoencoder(auto_encoder)
        print(auto_encoder.summary())
        return auto_encoder