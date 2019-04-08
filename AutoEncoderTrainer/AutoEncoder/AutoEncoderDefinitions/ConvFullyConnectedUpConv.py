from keras import Input, Model
from keras.callbacks import TensorBoard
from keras.layers import Conv2D, Flatten, Dense
from keras.optimizers import Adam

from AutoEncoderTrainer.AutoEncoder.AutoEncoder import AutoEncoder


class ConvFullyConnectedUpConv(AutoEncoder):
    def get_num_decoding_layers_to_rip_out(self):
        pass

    def __init__(self, model_hyper_parameters):
        AutoEncoder.__init__(self, model_hyper_parameters)

    def create_autoencoder(self, input_image_vector):
        flattened_vector_size = self.image_width_after_rescale * self.image_height_after_rescale * self.image_depth_after_rescale
        input_image = Input(shape=(self.image_width_after_rescale,
                                   self.image_height_after_rescale,
                                   self.image_depth_after_rescale))
        conv1 = Conv2D(128, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(input_image)
        fully_connected_encoded = Flatten()(conv1)
        fully_connected_encoded = Dense(flattened_vector_size,
                                        activation='linear',
                                        kernel_initializer=self.get_initializer())(fully_connected_encoded)
        # fully_connected_encoded = Reshape((-1, self.image_width_after_rescale,
        #                                    self.image_height_after_rescale))(fully_connected_encoded)
        # up_conv_decoded = Conv2D(128, (3, 3),
        #                          activation='linear',
        #                          padding='same',
        #                          kernel_initializer='he_normal')(UpSampling2D(size=(3, 3))(fully_connected_encoded))
        # decoded = Conv2D(1, 1, activation='linear')(up_conv_decoded)

        model = Model(input=input_image, output=fully_connected_encoded)

        model.compile(optimizer=Adam(lr=1e-4), loss='binary_crossentropy', metrics=['accuracy'])

        return model

    def train(self, input_matrix, model, validation_matrix):
        model.fit(input_matrix, input_matrix,
                  epochs=self.hyper_params.number_of_epochs_for_training,
                  batch_size=self.hyper_params.batch_size,
                  shuffle=True,
                  validation_data=(validation_matrix, validation_matrix),
                  callbacks=[TensorBoard(log_dir=self.hyper_params.tensor_board_directory,
                                         histogram_freq=0,
                                         write_graph=True,
                                         write_images=False)])
        self.save_model(model)
        return model


