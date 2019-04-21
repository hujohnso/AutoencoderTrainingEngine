import json
import os
import pickle
import shutil

import cv2
from keras.engine.saving import load_model


class ResultsWriter:
    parameters = None
    file_to_print_parameters_to = "parameters.json"
    file_to_print_model_history_to = "loss_history.txt"
    model = None
    root_file_path = None
    model = None

    def __init__(self, model_hyper_parameters):
        self.parameters = model_hyper_parameters
        self.root_file_path = self.parameters.base_file_path + "Results/" + self.parameters.results_folder
        self.load_old_model_if_present()
        self.delete_folder_and_create_new_empty_folder(self.root_file_path)
        self.delete_folder_and_create_new_empty_folder(self.root_file_path + "/images")
        self.save_model_if_present()

    def load_old_model_if_present(self):
        if self.parameters.load_model:
            self.model = load_model(self.parameters.working_model_path + self.parameters.model_name)

    def save_model_if_present(self):
        if self.parameters.load_model:
            self.model.save(self.parameters.working_model_path + self.parameters.model_name)

    def write_hyper_parameters_to_file(self):
        with open(self.root_file_path + "/" + self.file_to_print_parameters_to, 'w') as output:
            json.dump(self.parameters.__dict__, output, sort_keys=True)

    def write_model_history_to_file(self):
        with open(self.root_file_path + "/" + self.file_to_print_model_history_to, 'w') as output:
            output.write('The final loss for this model was: ' + repr(self.model.history.history['loss'][-1]) + '\n')
            output.write('val_loss/The validation loss: \n')
            output.write(repr(self.model.history.history['val_loss']) + '\n')
            output.write('loss/The training loss: \n')
            output.write(repr(self.model.history.history['loss']) + '\n')

    def write_image_matrix_to_files(self, image_matrix, name):
        i = 0
        for image in image_matrix:
            cv2.imwrite(self.root_file_path + "/images/" + name + "%d.jpg" % i, image)
            i += 1

    def write_all_information(self, model, training_matrix_orig, training_matrix, validation_matrix_orig, validation_matrix):
        if self.parameters.train_autoencoder:
            self.model = model
            self.write_hyper_parameters_to_file()
            self.write_model_history_to_file()
            self.write_image_matrix_to_files(training_matrix_orig, 'training_matrix_original')
            self.write_image_matrix_to_files(training_matrix, 'training_matrix')
            self.write_image_matrix_to_files(validation_matrix_orig, 'validation_matrix_original')
            self.write_image_matrix_to_files(validation_matrix, 'validation_matrix')

    def delete_folder_and_create_new_empty_folder(self, file_path):
        if os.path.isdir(file_path):
            os.chmod(file_path, 0o777)
            shutil.rmtree(file_path)
        os.makedirs(file_path)