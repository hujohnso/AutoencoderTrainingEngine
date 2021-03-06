from keras.callbacks import TensorBoard

from AutoEncoderTrainer.AutoEncoder.AutoEncoderDefinitions.ConvAutoEncoder import ConvAutoEncoder
from AutoEncoderTrainer.AutoEncoderConverter import AutoEncoderConverter
from AutoEncoderTrainer.AutoEncoderTrainer import AutoEncoderTrainer
from AutoEncoderTrainer.ModelRunner.ModelHyperParameters import ModelHyperParametersSimpleAnimationColor
from Segmenter.SegmentLabelRunner import SegmentLabelRunner
from SmartTrainer.SmartTrainer import SmartTrainer


class StateObjectClassifier:
    def __init__(self, model_hyper_parameters, auto_encoder_object):
        self.model_hyper_parameters = model_hyper_parameters
        self.auto_encoder_object = auto_encoder_object

    def train_new_model_on_state_objects(self):
        auto_encoder_model = self.get_trained_auto_encoder_model()
        print("You have finished getting the trained auto encoder \n")
        state_object_classes = self.get_state_object_information(self.model_hyper_parameters)
        print("You have finished getting the state object information")
        classification_model = self.get_network_for_fine_tuned_classification(auto_encoder_model,
                                                                              self.auto_encoder_object,
                                                                              state_object_classes.number_of_objects_identified)
        print("You just finished training your classification model!")
        return self.train_classification_model(classification_model, state_object_classes)

    def get_trained_auto_encoder_model(self):
        auto_encoder_trainer = AutoEncoderTrainer()
        if self.model_hyper_parameters.load_model_for_state_object_training:
            auto_encoder_model = auto_encoder_trainer.load_model_with_weights(self.auto_encoder_object)
        else:
            auto_encoder_model = auto_encoder_trainer.run_all_steps(self.auto_encoder_object,
                                                                    self.model_hyper_parameters)
        return auto_encoder_model

    def get_state_object_information(self, model_hyper_parameters):
        segment_label_runner = SegmentLabelRunner(model_hyper_parameters.file_path_for_validation_set,
                                                  model_hyper_parameters.results_folder)
        segment_label_runner.run_segment_label_runner()
        return segment_label_runner.get_segment_labels()

    def get_network_for_fine_tuned_classification(self, auto_encoder_model, auto_encoder_object,
                                                  number_of_objects_to_expect):
        if auto_encoder_object.get_num_decoding_layers_to_rip_out() is None:
            raise Exception("You may not have set the number of layers to rip out in your autoencoder")
        auto_encoder_converter = AutoEncoderConverter(auto_encoder_model,
                                                      auto_encoder_object.get_num_decoding_layers_to_rip_out(),
                                                      number_of_objects_to_expect)
        return auto_encoder_converter.convert_autoencoder_into_object_classifier()

    def single_time_train(self, input_matrix, output_matrix, model, validation_input, validation_output, hyper_params):
        model.fit(input_matrix,
                  output_matrix,
                  epochs=self.model_hyper_parameters.number_of_epochs_for_fine_tuning,
                  batch_size=self.model_hyper_parameters.batch_size_for_fine_tuning,
                  shuffle=True,
                  callbacks=[TensorBoard(log_dir=self.model_hyper_parameters.tensor_board_directory,
                                         histogram_freq=0,
                                         write_graph=True,
                                         write_images=False)])
        self.save_model(model)
        return model

    def train_classification_model(self, classification_model, state_object_classes):
        if self.model_hyper_parameters.use_smart_train:
            smart_trainer = SmartTrainer()
            return smart_trainer.smart_train(state_object_classes.array_of_state_object_images,
                                             state_object_classes.array_of_state_object_classes,
                                             classification_model, None, None, self.model_hyper_parameters,
                                             lambda a, b, c, d, e, f: self.single_time_train(a, b, c, d, e, f))
        else:
            return self.single_time_train(state_object_classes.array_of_state_object_images,
                                          state_object_classes.array_of_state_object_classes, classification_model,
                                          None, None, self.model_hyper_parameters)

    def save_model(self, model_to_save):
        model_to_save.save(
            self.model_hyper_parameters.working_model_path + self.model_hyper_parameters.results_folder + "_fine_tuned.h5")
