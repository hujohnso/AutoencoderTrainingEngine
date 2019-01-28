from keras.layers import Conv2D, BatchNormalization, LeakyReLU, Add


def residual_block(self, y, nb_channels, _strides=(1, 1), _project_shortcut=False):
    shortcut = y

    # down-sampling is performed with a stride of 2
    y = Conv2D(nb_channels,
               kernel_size=(3, 3),
               strides=_strides,
               padding='same',
               kernel_initializer=self.get_initializer(),
               bias_initializer=self.get_bias_initializer())(y)
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