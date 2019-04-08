import cv2
from keras.engine.saving import load_model
from keras.initializers import RandomUniform
import abc
import matplotlib.pyplot as plt
import numpy
from keras.optimizers import Adam
from skimage.transform import rescale
from keras.callbacks import TensorBoard
import random as rand

from AutoEncoderTrainer.AutoEncoder.AutoEncoderDefinitions import ImageManipulationType


class AutoEncoder:
    hyper_params = None
    image_width_after_rescale: int = None
    image_height_after_rescale: int = None
    image_depth_after_rescale: int = None

    def __init__(self, model_hyper_parameters):
        self.hyper_params = model_hyper_parameters

    @staticmethod
    def get_regularizer():
        return None

    @staticmethod
    def get_initializer():
        return RandomUniform()

    @staticmethod
    def get_bias_initializer():
        return None
        # return RandomUniform(minval=.001, maxval=.1, seed=None)

    @abc.abstractmethod
    def create_autoencoder(self, input_image_vector):
        return

    @abc.abstractmethod
    def get_num_decoding_layers_to_rip_out(self):
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
        return self.init_image_matrix(self.hyper_params.file_path_for_training_set,
                                      self.hyper_params.number_of_images)

    def init_validation_matrix(self):
        return self.init_image_matrix(self.hyper_params.file_path_for_validation_set,
                                      self.hyper_params.number_of_images_for_validation)

    def init_image_matrix(self, folder_with_images, number_of_frames):
        matrix = None
        skipping_factor = int(len(cv2.os.listdir(folder_with_images)) / number_of_frames)
        num_in_matrix = 0
        image_number = 0
        for filename in cv2.os.listdir(folder_with_images):
            if not filename.endswith(".jpg") and not filename.endswith(".png"):
                continue
            if num_in_matrix == number_of_frames:
                break
            if image_number % skipping_factor == 0:
                image = cv2.imread(folder_with_images + "/" + filename, self.get_image_read_parameter())
                image = self.prepare_single_image(image)
                if num_in_matrix == 0:
                    self.save_original_dims(image)
                    matrix = numpy.empty([number_of_frames,
                                          self.image_width_after_rescale,
                                          self.image_height_after_rescale,
                                          self.image_depth_after_rescale])
                matrix[num_in_matrix, :, :, :] = image
                num_in_matrix += 1
            image_number += 1
        return matrix

    def get_image_read_parameter(self):
        im_read_color_parameter = None
        if self.hyper_params.as_gray:
            im_read_color_parameter = 0
        else:
            im_read_color_parameter = 1
        return im_read_color_parameter

    def prepare_single_image(self, image):
        image = self.rescaler(image)
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
        image = self.inverse_rescaler(image_matrix)
        return image

    def train(self, input_matrix, model, validation_matrix):
        model.fit(input_matrix, input_matrix.reshape(self.hyper_params.number_of_images, -1),
                  epochs=self.hyper_params.number_of_epochs_for_training,
                  batch_size=self.hyper_params.batch_size,
                  shuffle=True,
                  validation_data=(
                      validation_matrix,
                      validation_matrix.reshape(self.hyper_params.number_of_images_for_validation, -1)),
                  callbacks=[TensorBoard(log_dir=self.hyper_params.tensor_board_directory,
                                         histogram_freq=0,
                                         write_graph=True,
                                         write_images=False)])
        self.save_model(model)
        return model

    def save_model(self, model_to_save):
        model_to_save.save(self.hyper_params.working_model_path + self.hyper_params.model_name)

    def build_model(self, input_image_vector):
        if self.hyper_params.load_model:
            return load_model(self.hyper_params.working_model_path + self.hyper_params.model_name)
        else:
            return self.create_autoencoder(input_image_vector)

    def visualize(self, trained_model):
        i = 2
        fig, ax = plt.subplots(i, 3)
        for x in range(i):
            image = None
            if x < i * .5:
                image = self.get_random_image_for_visualize(self.hyper_params.file_path_for_training_set)
            else:
                image = self.get_random_image_for_visualize(self.hyper_params.file_path_for_validation_set)
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

    # Only works in greyscale for now
    def get_results_matrices(self, trained_model):
        number_of_result_images = 5
        original_matrix = numpy.empty([number_of_result_images,
                                       self.hyper_params.pixel_resize_for_visualize,
                                       self.hyper_params.pixel_resize_for_visualize])
        results_matrix = numpy.empty([number_of_result_images,
                                      self.hyper_params.pixel_resize_for_visualize,
                                      self.hyper_params.pixel_resize_for_visualize])
        for x in range(number_of_result_images):
            image = self.get_random_image_for_visualize(self.hyper_params.file_path_for_training_set)
            original_matrix[x, :, :] = self.reformat_auto_encoder_format(self.prepare_single_image(image))
            results_matrix[x, :, :] = self.reformat_auto_encoder_format(
                trained_model.predict(self.prepare_single_image(image).reshape(1,
                                                                               self.image_width_after_rescale,
                                                                               self.image_height_after_rescale,
                                                                               self.image_depth_after_rescale)))
        return original_matrix, results_matrix

    def get_results_matrix_and_transform_input_matrix(self, trained_model, input_matrix):
        number_of_result_images = 5
        size_of_input_matrix = input_matrix.shape[0] - 1
        if self.hyper_params.as_gray:
            original_matrix = numpy.empty([number_of_result_images,
                                           self.hyper_params.pixel_resize_for_visualize,
                                           self.hyper_params.pixel_resize_for_visualize])
            results_matrix = numpy.empty([number_of_result_images,
                                          self.hyper_params.pixel_resize_for_visualize,
                                          self.hyper_params.pixel_resize_for_visualize])
        else:
            original_matrix = numpy.empty([number_of_result_images,
                                           self.hyper_params.pixel_resize_for_visualize,
                                           self.hyper_params.pixel_resize_for_visualize,
                                           self.image_depth_after_rescale])
            results_matrix = numpy.empty([number_of_result_images,
                                          self.hyper_params.pixel_resize_for_visualize,
                                          self.hyper_params.pixel_resize_for_visualize,
                                          self.image_depth_after_rescale])
        for x in range(number_of_result_images):
            image = input_matrix[rand.randint(0, size_of_input_matrix)]
            if self.hyper_params.as_gray:
                original_matrix[x, :, :] = self.reformat_auto_encoder_format(image)
                results_matrix[x, :, :] = self.reformat_auto_encoder_format(
                    trained_model.predict(image.reshape(1,
                                                        self.image_width_after_rescale,
                                                        self.image_height_after_rescale,
                                                        self.image_depth_after_rescale)))
            else:
                original_matrix[x, :, :, :] = self.reformat_auto_encoder_format(image)
                results_matrix[x, :, :, :] = self.reformat_auto_encoder_format(
                    trained_model.predict(image.reshape(1,
                                                        self.image_width_after_rescale,
                                                        self.image_height_after_rescale,
                                                        self.image_depth_after_rescale)))
        return original_matrix, results_matrix

    def get_random_image_for_visualize(self, folder_containing_images):
        list_of_images = cv2.os.listdir(folder_containing_images)
        image_name = ""
        while not image_name.endswith(".jpg"):
            image_name = list_of_images[rand.randint(0, len(list_of_images))]
        return cv2.imread(folder_containing_images + "/" + image_name, self.get_image_read_parameter())

    def rescaler(self, image_to_alter):
        image = None
        if self.hyper_params.type_of_transformation == ImageManipulationType.ImageManipulationType.PIXEL:
            image = cv2.resize(image_to_alter,
                               (self.hyper_params.pixel_resize_value, self.hyper_params.pixel_resize_value))
        elif self.hyper_params.type_of_transformation == ImageManipulationType.ImageManipulationType.RATIO:
            image = rescale(image_to_alter, self.hyper_params.image_rescale_value, anti_aliasing=False)
        elif self.hyper_params.type_of_transformation == ImageManipulationType.ImageManipulationType.NONE:
            image = image_to_alter
        image = image / 255
        return image

    def inverse_rescaler(self, image_to_alter):
        image_to_alter = image_to_alter * 255
        return cv2.resize(image_to_alter, (self.hyper_params.pixel_resize_for_visualize,
                                           self.hyper_params.pixel_resize_for_visualize))
        # image = rescale(image_matrix, 1.0 / self.hyper_params.image_rescale_value, anti_aliasing=False)

    def compile_autoencoder(self, auto_encoder):
        if self.hyper_params.adam_specify_learning_rate and not self.hyper_params.adam_decay_rate is None:
            auto_encoder.compile(optimizer=Adam(lr=self.hyper_params.adam_alpha,
                                                decay=self.hyper_params.adam_decay_rate),
                                 loss='mean_squared_error',
                                 metrics=['binary_crossentropy'])
        elif self.hyper_params.adam_specify_learning_rate and self.hyper_params.adam_decay_rate is None:
            auto_encoder.compile(optimizer=Adam(lr=self.hyper_params.adam_alpha),
                                 loss='mean_squared_error',
                                 metrics=['binary_crossentropy'])
        else:
            auto_encoder.compile(optimizer='Adam', loss='mean_squared_error', metrics=['binary_crossentropy'])

    def load_autoencoder(self):
        # global image_width_after_rescale
        # global image_height_after_rescale
        # global image_depth_after_rescale
        # self.image_width_after_rescale = self.hyper_params.pixel_resize_value
        # self.image_height_after_rescale = self.hyper_params.pixel_resize_value
        # if self.hyper_params.as_gray:
        #     self.image_depth_after_rescale = 1
        # else:
        #     self.image_depth_after_rescale = 3
        return self.build_model(None)
