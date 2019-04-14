import time

from AutoEncoderTrainer.Results import ResultsWriter


class AutoEncoderTrainer:
    def __init__(self):
        pass

    def timer(self, executable, function_executed):
        start_time = time.time()
        return_value = executable()
        end_time = time.time()
        print("It took ", end_time - start_time, " for ", function_executed, " to execute")
        return return_value

    def run_all_steps(self, auto_encoder_local, hyper_parameters_local):
        results_writer = ResultsWriter.ResultsWriter(hyper_parameters_local)
        input_matrix = self.timer(lambda: auto_encoder_local.init_training_matrix(), "training set creation")
        validation_matrix = self.timer(lambda: auto_encoder_local.init_validation_matrix(),
                                       "validation/dev set creation")
        auto_encoder_model = self.timer(lambda: auto_encoder_local.build_model(input_matrix), "model creation")
        auto_encoder_model = self.timer(
            lambda: auto_encoder_local.train(input_matrix, auto_encoder_model, validation_matrix, hyper_parameters_local), "the model to train")
        original, results = auto_encoder_local.get_results_matrix_and_transform_input_matrix(auto_encoder_model,
                                                                                             input_matrix)
        original_validation, results_validation = auto_encoder_local.get_results_matrix_and_transform_input_matrix(
            auto_encoder_model, validation_matrix)
        results_writer.write_all_information(auto_encoder_model, original, results, original_validation,
                                             results_validation)
        return auto_encoder_model

    def load_model_with_weights(self, auto_encoder_local):
        return auto_encoder_local.load_autoencoder()

