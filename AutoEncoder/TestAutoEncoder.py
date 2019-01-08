from keras.initializers import RandomUniform, Zeros, Constant, SparceInitializer
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Flatten, BatchNormalization, LeakyReLU, Add
from keras import regularizers
from keras.legacy import layers
from keras.models import Sequential, Model
import matplotlib.pyplot as plt
import time
import numpy
from skimage import data, img_as_float
from skimage.transform import rescale
from keras.callbacks import TensorBoard
import random as rand

filePath = "/home/hujohnso/Documents/Research2018/FrameExtractor/tmp/"
totalNumberOfFrames = 300
number_of_epochs_for_training = 1
batch_size = 300
image_width_after_rescale: int
image_height_after_rescale: int
image_depth = 3
image_rescale_value = 1.0 / 15.0
test_image_name = "Cute Baby Holland Lop Bunnies Playing Inside the House.mp4frame1700.jpg"
as_gray = False


def create_image_vector(file_path, number_of_images, original_video):
    inputMatrix = None
    for i in range(number_of_images):
        # image = img_as_float(data.load(file_path + original_video + "frame%d.jpg" % (300 + i * 10), as_gray=as_gray))
        image = img_as_float(data.load(file_path + original_video + "frame%d.jpg" % 300, as_gray=as_gray))
        image = prepare_single_image(image)
        input_dim = image.shape[1]
        if i == 0:
            inputMatrix = numpy.empty([number_of_images, input_dim])
        inputMatrix[i, :] = image
    return inputMatrix


def create_image_vector_conv(file_path, number_of_images, original_video):
    input_matrix = None
    for i in range(number_of_images):
        image = img_as_float(data.load(file_path + original_video + "frame%d.jpg" % (300 + i * 10), as_gray=as_gray))
        image = prepare_single_image_conv(image)
        if i == 0:
            input_matrix = numpy.empty([number_of_images,
                                        image.shape[0],
                                        image.shape[1],
                                        image.shape[2]])
        input_matrix[i, :, :, :] = image
    return input_matrix


# This will eventually be a convolution network but just to get everything
# going I am doing FC layers
def create_auto_encoder_fully_connected(input_vector):
    model = Sequential()
    model.add(
        Dense(200, activation='relu', input_dim=input_vector.shape[1]))
    model.add(Dense(512, activation='relu', activity_regularizer=regularizers.l1(10e-5)))
    model.add(Dense(512, activation='relu', activity_regularizer=regularizers.l1(10e-5)))
    model.add(Dense(512, activation='relu', activity_regularizer=regularizers.l1(10e-5)))
    model.add(Dense(512, activation='relu', activity_regularizer=regularizers.l1(10e-5)))
    model.add(Dense(512, activation='relu', activity_regularizer=regularizers.l1(10e-5)))
    model.add(Dense(512, activation='relu', activity_regularizer=regularizers.l1(10e-5)))
    model.add(Dense(input_vector.shape[1], activation='relu'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['binary_crossentropy'])
    return model

# Overall we can think of regularizers as adding a penality for the model being too complex.
#   To remember this think about ho


# Essentially what KL divergence does is it pentalizes terms for being too far from a particular value
# This is used on particular layers to enforce a particular neuron firing to a particular feature being seen.
# I want this because I want any given feature to fire for a particular object that the network thinks it is
# Then temporally when

# L1 regularization also known as Lasso regression: Lasso regression adds extra loss for weights that are not zero
# This is in an effort to reduce the number of activations that matter  Lasso does this by adding the the abs of the
# weights

# L2 regularization also known as Ridge regression: this is the same as L1 except it squares values.
def getRegularizer():
    return None


def get_initializer():
    return SparceInitializer()
    # return RandomUniform(minval=0, maxval=.3, seed=5)





def get_bias_initializer():
    return None
    # return RandomUniform(minval=.001, maxval=.1, seed=None)


def create_auto_encoder_conv(input_image_vector):
    input_image = Input(shape=(input_image_vector.shape[1],
                               input_image_vector.shape[2],
                               input_image_vector.shape[3]))
    # x = Conv2D(128, (3, 3), activation='relu', padding='same')(input_image)
    # x = MaxPooling2D((2, 2), padding='same')(x)
    # x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    # x = MaxPooling2D((2, 2), padding='same')(x)
    # x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    # encoded = MaxPooling2D((2, 2), padding='same')(x)
    # x = Conv2D(128, (3, 3), activation='relu', padding='same')(encoded)
    # x = MaxPooling2D((2, 2), padding='same')(x)
    # x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    # x = MaxPooling2D((2, 2), padding='same')(x)
    # x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    # x = MaxPooling2D((2, 2), padding='same')(x)
    # x = residual_block(input_image, 128, _strides=(3, 3))
    # x = residual_block(x, 246, _strides=(5, 5))
    # x = MaxPooling2D((2, 2), padding='same')(x)
    # x = residual_block(x, 512, _strides=(10, 10))
    # x = MaxPooling2D((4, 4), padding='same')(x)
    # x = residual_block(x, 1024, _strides=(3, 3))
    # x = MaxPooling2D((4, 4), padding='same')(x)
    # x = residual_block(x, 1024, _strides=(3, 3))
    # x = MaxPooling2D((4, 4), padding='same')(x)
    # x = residual_block(x, 1024, _strides=(3, 3))
    # x = MaxPooling2D((4, 4), padding='same')(x)
    # x = residual_block(x, 1024, _strides=(3, 3))
    # x = MaxPooling2D((4, 4), padding='same')(x)
    # decoded = Conv2D(1, (3, 3), activation='relu', padding='same')(x)
    decoded = Flatten()(input_image)
    # decoded = Dense(input_image_vector.shape[1] * input_image_vector.shape[2] * input_image_vector.shape[3],
    #                 activation='relu', kernel_initializer=get_initializer(), bias_initializer=get_bias_initializer(),
    #                 activity_regularizer= getRegularizer())(decoded)
    # decoded = Dense(input_image_vector.shape[1] * input_image_vector.shape[2] * input_image_vector.shape[3],
    #                 activation='relu', kernel_initializer=get_initializer(), bias_initializer=get_bias_initializer(),
    #                 activity_regularizer= getRegularizer())(decoded)
    decoded = Dense(input_image_vector.shape[1] * input_image_vector.shape[2] * input_image_vector.shape[3],
                    activation='relu', kernel_initializer=get_initializer(),
                    activity_regularizer=getRegularizer())(decoded)
    autoEncoder = Model(input_image, decoded)
    autoEncoder.compile(optimizer='adam', loss='mean_squared_error', metrics=['binary_crossentropy'])
    return autoEncoder


def residual_block(y, nb_channels, _strides=(1, 1), _project_shortcut=False):
    shortcut = y

    # down-sampling is performed with a stride of 2
    y = Conv2D(nb_channels, kernel_size=(3, 3), strides=_strides, padding='same', kernel_initializer=get_initializer(),
               bias_initializer=get_bias_initializer())(y)
    # y = MaxPooling2D((2, 2), padding='same')(y)
    y = BatchNormalization()(y)
    y = LeakyReLU()(y)

    y = Conv2D(nb_channels, kernel_size=(3, 3), strides=(1, 1), padding='same', kernel_initializer=get_initializer(),
               bias_initializer=get_bias_initializer())(y)
    # y = MaxPooling2D((2, 2), padding='same')(y)
    y = BatchNormalization()(y)

    # identity shortcuts used directly when the input and output are of the same dimensions
    if _project_shortcut or _strides != (1, 1):
        # when the dimensions increase projection shortcut is used to match dimensions (done by 1×1 convolutions)
        # when the shortcuts go across feature maps of two sizes, they are performed with a stride of 2
        shortcut = Conv2D(nb_channels, kernel_size=(1, 1), strides=_strides, padding='same', kernel_initializer=get_initializer(),
                          bias_initializer=get_bias_initializer())(shortcut)
        # y = MaxPooling2D((2, 2), padding='same')(y)
        shortcut = BatchNormalization()(shortcut)

    y = Add()([shortcut, y])
    y = LeakyReLU()(y)

    return y


# Note epochs is the number of times that the data is passed through the model for training
# A batch is how many training examples are feed through the model at once
def train_auto_encoder(input_vector, model):
    model.fit(input_vector,
              input_vector,
              epochs=number_of_epochs_for_training,
              batch_size=batch_size,
              shuffle=True,
              verbose=1
              )
    return model


def train_conv_auto_encoder(input_matrix, model):
    model.fit(input_matrix, input_matrix.reshape(totalNumberOfFrames, -1),
                    epochs=number_of_epochs_for_training,
                    batch_size=batch_size,
                    shuffle=True,
                    validation_data=(input_matrix, input_matrix.reshape(totalNumberOfFrames, -1)),
                    callbacks=[TensorBoard(log_dir='/tmp/autoencoder')])
    return model


# Here we will regularize the data because that is just kind of what we do
def prepare_single_image(image):
    global image_height_after_rescale
    global image_width_after_rescale
    image = rescale(image, image_rescale_value, anti_aliasing=False)
    image_height_after_rescale = image.shape[0]
    image_width_after_rescale = image.shape[1]
    image = image.flatten().reshape(1, -1)
    input_vector = image.astype('float32') / 255
    return input_vector


def inverse_prepare_single_image(image_vector):
    image_vector = image_vector * 255
    if as_gray:
        image = image_vector.reshape(image_height_after_rescale, image_width_after_rescale)
    else:
        image = image_vector.reshape(image_height_after_rescale, image_width_after_rescale, image_depth)
    image = rescale(image, 1.0 / image_rescale_value, anti_aliasing=False)
    return image


def prepare_single_image_conv(image):
    global image_height_after_rescale
    global image_width_after_rescale
    image = rescale(image, image_rescale_value, anti_aliasing=False)
    image_height_after_rescale = image.shape[0]
    image_width_after_rescale = image.shape[1]
    # image = image.astype('float32') / 255
    return image


def inverse_prepare_single_image_conv(image):
    image = image.reshape(image_height_after_rescale, image_width_after_rescale, 3)
    # image = rescale(image, 1.0 / image_rescale_value, anti_aliasing=False)
    # image = image * 255
    return image


def input_image_into_trained_model_and_visualize(trained_model, input_image_name):
    image = img_as_float(data.load(filePath + input_image_name, as_gray=as_gray))
    fig, axes = plt.subplots(1, 2)
    ax = axes.flatten()
    ax[0].set_title("Original Image", fontsize=12)
    ax[0].imshow(image)
    ax[0].set_axis_off()
    ax[1].set_title("Image after auto encoder", fontsize=12)
    image_after_encoder = inverse_prepare_single_image(trained_model.predict(prepare_single_image(image)))
    ax[1].imshow(image_after_encoder)
    ax[1].set_axis_off()
    fig.tight_layout()
    plt.show()


def input_image_into_trained_conv_auto_encoder_and_visualize(trained_model, input_image_name):
    image = img_as_float(data.load(filePath + input_image_name, as_gray=as_gray))
    fig, ax = plt.subplots(1, 2)
    ax[0].set_title("Original Image", fontsize=12)
    ax[0].imshow(image)
    ax[0].set_axis_off()
    ax[1].set_title("Image after auto encoder", fontsize=12)
    image_after_encoder = inverse_prepare_single_image_conv(
        trained_model.predict(numpy.expand_dims(prepare_single_image_conv(image), axis=0)))
    ax[1].imshow(image_after_encoder)
    ax[1].set_axis_off()
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    start_time = time.time()
    input_matrix = create_image_vector_conv(filePath, totalNumberOfFrames,
                                       "Cute Baby Holland Lop Bunnies Playing Inside the House.mp4")
    end_time = time.time()
    print("The time taken to load ", totalNumberOfFrames, " frames was all the photos is: ",
          (end_time - start_time), " seconds")
    auto_encoder_model = create_auto_encoder_conv(input_matrix)
    start_time = time.time()
    auto_encoder_model = train_conv_auto_encoder(input_matrix, auto_encoder_model)
    end_time = time.time()
    print("The time taken to train is: ",
          (end_time - start_time), " seconds")
    input_image_into_trained_conv_auto_encoder_and_visualize(auto_encoder_model, test_image_name)












