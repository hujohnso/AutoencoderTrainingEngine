from keras.optimizers import Adam

from AutoEncoder.AutoEncoder import AutoEncoder
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Flatten, BatchNormalization, LeakyReLU, Add
from keras import regularizers
from keras.legacy import layers
from keras.models import Sequential, Model


class FullyConnectedAutoEncoder(AutoEncoder):
    def __init__(self, model_hyper_parameters):
        AutoEncoder.__init__(self, model_hyper_parameters)

    def create_autoencoder(self, input_image_vector):
        flattened_vector_size = self.image_width_after_rescale * self.image_height_after_rescale * self.image_depth_after_rescale
        input_image = Input(shape=(self.image_width_after_rescale,
                                   self.image_height_after_rescale,
                                   self.image_depth_after_rescale))
        encoded = Flatten()(input_image)
        encoded = Dense(flattened_vector_size,
                        activation='linear', kernel_initializer=self.get_initializer())(encoded)
        encoded = Dense(int (flattened_vector_size * .15),
                        activation= 'linear',
                        kernel_initializer= self.get_initializer())(encoded)
        decoded = Dense(flattened_vector_size,
                        activation='linear', kernel_initializer=self.get_initializer())(encoded)
        auto_encoder = Model(input_image, decoded)
        if self.hyper_params.adam_specify_learning_rate:
            auto_encoder.compile(optimizer=Adam(lr=self.hyper_params.adam_specify_learning_rate,
                                                decay=self.hyper_params.adam_specify_learning_rate),
                                 loss='mean_squared_error',
                                 metrics=['binary_crossentropy'])
        else:
            auto_encoder.compile(optimizer='Adam', loss='mean_squared_error', metrics=['binary_crossentropy'])
        print(auto_encoder.summary())
        return auto_encoder
