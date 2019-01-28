import cv2
from keras.engine.saving import load_model
from keras.initializers import RandomUniform, Zeros, Constant, SparceInitializer
import abc
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


class AutoEncoder:
    hyper_params = None
    image_width_after_rescale: int = None
    image_height_after_rescale: int = None
    image_depth_after_rescale: int = None

    def __init__(self, model_hyper_parameters):
        self.hyper_params = model_hyper_parameters

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
    @staticmethod
    def get_regularizer():
        return None

    @staticmethod
    def get_initializer():
        # return SparceInitializer()
        return RandomUniform()

    @staticmethod
    def get_bias_initializer():
        return None
        # return RandomUniform(minval=.001, maxval=.1, seed=None)

    @abc.abstractmethod
    def create_auto_encoder(self, input_image_vector):
        return

    def save_original_dims(self, input_image):
        global image_width_after_rescale
        global image_height_after_rescale
        global image_depth_after_rescale
        self.image_width_after_rescale = input_image.shape[0]
        self.image_height_after_rescale = input_image.shape[1]
        if self.hyper_params.as_gray:
            self.image_depth_after_rescale = 1
        else:
            self.image_depth_after_rescale = input_image.shape[2]

    def init_training_matrix(self):
        input_matrix = None
        for i in range(self.hyper_params.number_of_images):
            image = cv2.imread(self.hyper_params.file_path_for_frames + self.hyper_params.original_video + "frame%d.jpg" % (300 + i * self.hyper_params.frame_skipping_factor), 0)
            image = self.prepare_single_image(image)
            if i == 0:
                self.save_original_dims(image)
                input_matrix = numpy.empty([self.hyper_params.number_of_images,
                                            self.image_width_after_rescale,
                                            self.image_height_after_rescale,
                                            self.image_depth_after_rescale])
            input_matrix[i, :, :, :] = image
        return input_matrix

    def prepare_single_image(self, image):
        (thresh, image) = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        image = rescale(image, self.hyper_params.image_rescale_value, anti_aliasing=False)
        if self.image_width_after_rescale is None:
            self.save_original_dims(image)
        return image.reshape(self.image_width_after_rescale,
                             self.image_height_after_rescale,
                             self.image_depth_after_rescale)

    def prepare_single_image_for_visualize(self, image):
        if self.hyper_params.as_gray:
            return image.reshape(self.image_width_after_rescale, self.image_height_after_rescale)
        else:
            return image

    def reformat_auto_encoder_format(self, image_vector):
        image_matrix = image_vector.reshape(self.image_width_after_rescale,
                                            self.image_height_after_rescale,
                                            self.image_depth_after_rescale)
        image_matrix = self.prepare_single_image_for_visualize(image_matrix)
        image = rescale(image_matrix, 1.0 / self.hyper_params.image_rescale_value, anti_aliasing=False)
        return image

    def train(self, input_matrix, model):
        model.fit(input_matrix, input_matrix.reshape(self.hyper_params.number_of_images, -1),
                  epochs=self.hyper_params.number_of_epochs_for_training,
                  batch_size=self.hyper_params.batch_size,
                  shuffle=True,
                  validation_data=(input_matrix, input_matrix.reshape(self.hyper_params.number_of_images, -1)),
                  callbacks=[TensorBoard(log_dir=self.hyper_params.tensor_board_directory,
                                         histogram_freq=0,
                                         write_graph=True,
                                         write_images=False)])
        self.save_model(model)
        return model

    def save_model(self, model_to_save):
        model_to_save.save(self.hyper_params.working_model_path)

    def build_model(self, input_image_vector):
        if self.hyper_params.load_model:
            return load_model(self.hyper_params.working_model_path)
        else:
            return self.create_auto_encoder(input_image_vector)

    def visualize(self, trained_model):
        i = 3
        fig, ax = plt.subplots(3, 3)
        for x in range(i):
            # image = img_as_float((
            #     data.load(
            #         self.hyper_params.file_path_for_frames +
            #         self.hyper_params.original_video +
            #         "frame%d.jpg" % (300 + x * 100),
            #         as_gray=self.hyper_params.as_gray)))
            image = cv2.imread(self.hyper_params.file_path_for_frames + self.hyper_params.original_video + "frame%d.jpg" % (300 + x * 100), 0)
            ax[x][0].set_title("Original Image", fontsize=12)
            ax[x][0].imshow(image)
            ax[x][0].set_axis_off()

            ax[x][1].set_title("image fed in", fontsize=12)
            ax[x][1].imshow(self.reformat_auto_encoder_format(self.prepare_single_image(image)))
            ax[x][1].set_axis_off()

            ax[x][2].set_title("Image after auto encoder", fontsize=12)
            image_after_encoder = self.reformat_auto_encoder_format(
                trained_model.predict(self.prepare_single_image(image).reshape(1,
                                                                               self.image_width_after_rescale,
                                                                               self.image_height_after_rescale,
                                                                               self.image_depth_after_rescale)))
            ax[x][2].imshow(image_after_encoder)
            ax[x][2].set_axis_off()
        fig.tight_layout()
        plt.show()

























