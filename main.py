import time

from AutoEncoderTrainer.AutoEncoder.AutoEncoderDefinitions import MNISTExampleAutoEncoder, \
    FullyConnectedAutoEncoderExponential, FullyConnectedAutoEncoderHyperbolicTangent, FullyConnectedAutoEncoder, \
    FullyConnectedAutoEncoderDeep
from AutoEncoderTrainer.AutoEncoder.AutoEncoderDefinitions.ConvAutoEncoder import ConvAutoEncoder
from AutoEncoderTrainer.AutoEncoderConverter import AutoEncoderConverter
from AutoEncoderTrainer.AutoEncoderTrainer import AutoEncoderTrainer
from AutoEncoderTrainer.ModelRunner.ModelHyperParameters import ModelHyperParametersRealImagesColor, \
    ModelHyperParametersRealImagesGray, ModelHyperParametersSimpleAnimationColor

# hyper_parameters = ModelHyperParameters()

# hyper_parameters = ModelHyperParametersRealImagesGray()
# hyper_parameters = ModelHyperParametersMNIST()
from StateObjectClassifier.StateObjectClassifier import StateObjectClassifier

hyper_parameters = ModelHyperParametersRealImagesColor()
# hyper_parameters = ModelHyperParametersAnimationGrey()
# auto_encoder = ConvFullyConnectedUpConv(hyper_parameters)
# auto_encoder = FullyConnectedAutoEncoder.FullyConnectedAutoEncoder(hyper_parameters)
# auto_encoder = AlexNetConvolutionalAutoEncoder(hyper_parameters)
# auto_encoder = ConvAutoEncoder.ConvAutoEncoder(hyper_parameters)
# auto_encoder = Unet()
auto_encoder = MNISTExampleAutoEncoder.MNISTExampleAutoEncoder(hyper_parameters)


def timer(executable, function_executed):
    start_time = time.time()
    return_value = executable()
    end_time = time.time()
    print("It took ", end_time - start_time, " for ", function_executed, " to execute")
    return return_value


def run_number_of_images_experiment(trainer):
    for i in range(25):
        hyper_parameters.number_of_images = (i + 1) * 20
        hyper_parameters.number_of_images_for_validation = int(hyper_parameters.number_of_images * .2)
        hyper_parameters.model_name = "real_images_grey_5_%d.h5" % hyper_parameters.number_of_images
        hyper_parameters.results_folder = "real_images_grey_5_%d" % hyper_parameters.number_of_images
        hyper_parameters.batch_size = hyper_parameters.number_of_images
        trainer.run_all_steps(auto_encoder, hyper_parameters)


def run_number_of_images_experiments(hyper_parameters_local, auto_encoder_local, name, trainer):
    for i in range(3):
        hyper_parameters_local.number_of_images = (i + 1) * 100
        hyper_parameters_local.number_of_images_for_validation = int(hyper_parameters_local.number_of_images * .2)
        hyper_parameters_local.model_name = name + "%d.h5" % hyper_parameters_local.number_of_images
        hyper_parameters_local.results_folder = name + "%d" % hyper_parameters_local.number_of_images
        hyper_parameters_local.batch_size = hyper_parameters_local.number_of_images
        trainer.run_all_steps(auto_encoder_local, hyper_parameters_local)


def run_activation_function_tests():
    hyper_parameters_local = ModelHyperParametersRealImagesGray()
    auto_encoder_local = FullyConnectedAutoEncoderExponential.FullyConnectedAutoEncoderExponential(
        hyper_parameters_local)
    run_number_of_images_experiments(hyper_parameters_local, auto_encoder_local, "fully_connected_elu_")
    auto_encoder_local = FullyConnectedAutoEncoderHyperbolicTangent.FullyConnectedAutoEncoderHyperbolicTangent(
        hyper_parameters_local)
    run_number_of_images_experiments(hyper_parameters_local, auto_encoder_local, "fully_connected_hyperbolic_tangent_")
    auto_encoder_local = FullyConnectedAutoEncoder.FullyConnectedAutoEncoder(hyper_parameters_local)
    run_number_of_images_experiments(hyper_parameters_local, auto_encoder_local, "fully_connected_linear_")


def run_deep_fully_connected_test():
    hyper_parameters_local = ModelHyperParametersRealImagesGray()
    auto_encoder_local = FullyConnectedAutoEncoderDeep.FullyConnectedAutoEncoderDeep(hyper_parameters_local)
    run_number_of_images_experiments(hyper_parameters_local, auto_encoder_local, "fully_connected_deep_")


# if __name__ == "__main__":
#     auto_encoder_trainer = AutoEncoderTrainer()
#     auto_encoder_trainer.run_all_steps(auto_encoder, hyper_parameters)

if __name__ == "__main__":
    model_hyper_parameters = ModelHyperParametersSimpleAnimationColor()
    auto_encoder = ConvAutoEncoder(model_hyper_parameters)
    state_object_classifier = StateObjectClassifier(model_hyper_parameters, auto_encoder)
    state_object_classifier.train_new_model_on_state_objects()