from keras import Input, Model
from keras.callbacks import TensorBoard
from keras.layers import Conv2D, MaxPooling2D, Dropout, UpSampling2D, concatenate
from keras.optimizers import Adam
from tensorflow.python.keras import initializers
from AutoEncoder.AutoEncoder import AutoEncoder


class FullyConvolutionalAutoEncoder(AutoEncoder):
    def __init__(self, model_hyper_parameters):
        AutoEncoder.__init__(self, model_hyper_parameters)

    def create_auto_encoder(self, input_image_vector):
        input_image = Input(shape=(self.image_width_after_rescale,
                                   self.image_height_after_rescale,
                                   self.image_depth_after_rescale))

        conv1 = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(input_image)
        conv1 = Conv2D(64, (3, 3), activation='relu', padding='same',  kernel_initializer='he_normal')(conv1)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

        conv2 = Conv2D(128, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
        conv2 = Conv2D(128, (3, 3), activation = 'relu', padding='same',  kernel_initializer='he_normal')(conv2)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

        conv3 = Conv2D(256, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
        conv3 = Conv2D(256, (3, 3), activation = 'relu', padding='same',  kernel_initializer='he_normal')(conv3)
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

        conv4 = Conv2D(512, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
        conv4 = Conv2D(512, (3, 3), activation = 'relu', padding='same',  kernel_initializer='he_normal')(conv4)
        drop4 = Dropout(0.5)(conv4)
        pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

        conv5 = Conv2D(1024, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
        conv5 = Conv2D(1024, (3, 3), activation = 'relu', padding='same',  kernel_initializer='he_normal')(conv5)
        drop5 = Dropout(0.5)(conv5)

        up6 = Conv2D(512, (2, 2), activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(drop5))
        merge6 = concatenate([drop4, up6], axis=3)
        conv6 = Conv2D(512, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
        conv6 = Conv2D(512, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(conv6)

        up7 = Conv2D(256, (2, 2), activation='relu', padding='same', kernel_initializer='he_normal')(
            UpSampling2D(size=(2, 2))(conv6))
        merge7 = concatenate([conv3, up7], axis=3)
        conv7 = Conv2D(256, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
        conv7 = Conv2D(256, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(conv7)

        up8 = Conv2D(128, (2, 2), activation='relu', padding='same', kernel_initializer='he_normal')(
            UpSampling2D(size=(2, 2))(conv7))
        merge8 = concatenate([conv2, up8], axis=3)
        conv8 = Conv2D(128, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
        conv8 = Conv2D(128, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(conv8)

        up9 = Conv2D(64, (2, 2), activation='relu', padding='same', kernel_initializer='he_normal')(
            UpSampling2D(size=(2, 2))(conv8))
        merge9 = concatenate([conv1, up9], axis=3)
        conv9 = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(merge9)
        conv9 = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
        conv9 = Conv2D(2, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
        conv10 = Conv2D(1, 1, activation='sigmoid')(conv9)

        model = Model(input=input_image, output=conv10)

        model.compile(optimizer=Adam(lr=1e-4), loss='binary_crossentropy', metrics=['accuracy'])

        return model

    def train(self, input_matrix, model):
        model.fit(input_matrix, input_matrix,
                  epochs=self.hyper_params.number_of_epochs_for_training,
                  batch_size=self.hyper_params.batch_size,
                  shuffle=True,
                  validation_data=(input_matrix, input_matrix),
                  callbacks=[TensorBoard(log_dir=self.hyper_params.tensor_board_directory,
                                         histogram_freq=0,
                                         write_graph=True,
                                         write_images=False)])
        self.save_model(model)
        return model


