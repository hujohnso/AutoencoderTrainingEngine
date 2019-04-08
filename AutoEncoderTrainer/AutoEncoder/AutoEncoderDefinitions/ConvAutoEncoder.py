
from keras.layers import Input, Dense, Conv2D, Flatten
from keras.models import Model


#
from AutoEncoderTrainer.AutoEncoder.AutoEncoder import AutoEncoder


class ConvAutoEncoder(AutoEncoder):
    def get_num_decoding_layers_to_rip_out(self):
        return 2

    def __init__(self, model_hyper_parameters):
        AutoEncoder.__init__(self, model_hyper_parameters)

    def create_autoencoder(self, input_image_vector):
        flattened_vector_size = self.image_width_after_rescale * self.image_height_after_rescale * self.image_depth_after_rescale
        input_image = Input(shape=(self.image_width_after_rescale,
                                   self.image_height_after_rescale,
                                   self.image_depth_after_rescale))
        encoded = Conv2D(16, (5, 5), activation='relu', padding='same', kernel_initializer=self.get_initializer())(input_image)
        encoded = Flatten()(encoded)
        decoded = Dense(64,
                        activation='relu', kernel_initializer=self.get_initializer(),
                        activity_regularizer=self.get_regularizer())(encoded)
        decoded = Dense(flattened_vector_size, activation='relu', kernel_initializer=self.get_initializer(), activity_regularizer=self.get_regularizer())(decoded)
        auto_encoder = Model(input_image, decoded)
        auto_encoder.compile(optimizer='adam', loss='mean_squared_error', metrics=['binary_crossentropy'])
        print(auto_encoder.summary())
        return auto_encoder
