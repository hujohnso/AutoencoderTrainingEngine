from abc import ABC

from keras.layers import Input, Dense, Flatten
from keras.models import Model

from AutoEncoderTrainer.AutoEncoder.AutoEncoder import AutoEncoder


class FullyConnectedAutoEncoderHyperbolicTangent(AutoEncoder, ABC):
    def __init__(self, model_hyper_parameters):
        AutoEncoder.__init__(self, model_hyper_parameters)

    def create_autoencoder(self, input_image_vector):
        flattened_vector_size = self.image_width_after_rescale * self.image_height_after_rescale * self.image_depth_after_rescale
        input_image = Input(shape=(self.image_width_after_rescale,
                                   self.image_height_after_rescale,
                                   self.image_depth_after_rescale))
        encoded = Flatten()(input_image)
        encoded = Dense(flattened_vector_size,
                        activation='tanh', kernel_initializer=self.get_initializer())(encoded)
        encoded = Dense(int(flattened_vector_size * .3),
                        activation='tanh',
                        kernel_initializer=self.get_initializer())(encoded)
        decoded = Dense(flattened_vector_size,
                        activation='tanh', kernel_initializer=self.get_initializer())(encoded)
        auto_encoder = Model(input_image, decoded)
        self.compile_autoencoder(auto_encoder)
        print(auto_encoder.summary())
        return auto_encoder