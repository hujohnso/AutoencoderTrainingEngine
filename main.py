import time

from AutoEncoder import FullyConnectedAutoEncoder, MNISTExampleAutoEncoder
from AutoEncoder.AlexNetConvolutionalAutoEncoder import AlexNetConvolutionalAutoEncoder
from AutoEncoder.ConvFullyConnectedUpConv import ConvFullyConnectedUpConv
from AutoEncoder.FullyConnectedAutoEncoderDeep import FullyConnectedAutoEncoderDeep
from AutoEncoder.FullyConnectedAutoEncoderExponential import FullyConnectedAutoEncoderExponential
from AutoEncoder.FullyConnectedAutoEncoderHyperbolicTangent import FullyConnectedAutoEncoderHyperbolicTangent
from ModelRunner.ModelHyperParameters import ModelHyperParametersRealImagesColor, ModelHyperParametersRealImagesGray, \
    ModelHyperParametersMNIST

# hyper_parameters = ModelHyperParameters()
from Results import ResultsWriter

# hyper_parameters = ModelHyperParametersRealImagesGray()
# hyper_parameters = ModelHyperParametersMNIST()
hyper_parameters = ModelHyperParametersRealImagesColor()
#hyper_parameters = ModelHyperParametersAnimationGrey()
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


def run_number_of_images_experiment():
    for i in range(25):
        hyper_parameters.number_of_images = (i + 1) * 20
        hyper_parameters.number_of_images_for_validation = int(hyper_parameters.number_of_images * .2)
        hyper_parameters.model_name = "real_images_grey_5_%d.h5" % hyper_parameters.number_of_images
        hyper_parameters.results_folder = "real_images_grey_5_%d" % hyper_parameters.number_of_images
        hyper_parameters.batch_size = hyper_parameters.number_of_images
        run_all_steps(auto_encoder, hyper_parameters)

def run_number_of_images_experiments(hyper_parameters_local, auto_encoder_local, name):
    for i in range(3):
        hyper_parameters_local.number_of_images = (i + 1) * 100
        hyper_parameters_local.number_of_images_for_validation = int(hyper_parameters_local.number_of_images * .2)
        hyper_parameters_local.model_name = name + "%d.h5" % hyper_parameters_local.number_of_images
        hyper_parameters_local.results_folder = name + "%d" % hyper_parameters_local.number_of_images
        hyper_parameters_local.batch_size = hyper_parameters_local.number_of_images
        run_all_steps(auto_encoder_local, hyper_parameters_local)


def run_activation_function_tests():
    hyper_parameters_local = ModelHyperParametersRealImagesGray()
    auto_encoder_local = FullyConnectedAutoEncoderExponential(hyper_parameters_local)
    run_number_of_images_experiments(hyper_parameters_local, auto_encoder_local, "fully_connected_elu_")
    auto_encoder_local = FullyConnectedAutoEncoderHyperbolicTangent(hyper_parameters_local)
    run_number_of_images_experiments(hyper_parameters_local, auto_encoder_local, "fully_connected_hyperbolic_tangent_")
    auto_encoder_local = FullyConnectedAutoEncoderHyperbolicTangent(hyper_parameters_local)
    run_number_of_images_experiments(hyper_parameters_local, auto_encoder_local, "fully_connected_hyperbolic_tangent_")
    auto_encoder_local = FullyConnectedAutoEncoder.FullyConnectedAutoEncoder(hyper_parameters_local)
    run_number_of_images_experiments(hyper_parameters_local, auto_encoder_local, "fully_connected_linear_")


def run_deep_fully_connected_test():
    hyper_parameters_local = ModelHyperParametersRealImagesGray()
    auto_encoder_local = FullyConnectedAutoEncoderDeep(hyper_parameters_local)
    run_number_of_images_experiments(hyper_parameters_local, auto_encoder_local, "fully_connected_deep_")


def run_all_steps(auto_encoder_local, hyper_parameters_local):
    results_writer = ResultsWriter.ResultsWriter(hyper_parameters_local)
    input_matrix = timer(lambda: auto_encoder_local.init_training_matrix(), "training set creation")
    validation_matrix = timer(lambda: auto_encoder_local.init_validation_matrix(), "validation/dev set creation")
    auto_encoder_model = timer(lambda: auto_encoder_local.build_model(input_matrix), "model creation")
    auto_encoder_model = timer(lambda: auto_encoder_local.train(input_matrix, auto_encoder_model, validation_matrix), "the model to train")
    original, results = auto_encoder_local.get_results_matrix_and_transform_input_matrix(auto_encoder_model, input_matrix)
    original_validation, results_validation = auto_encoder_local.get_results_matrix_and_transform_input_matrix(auto_encoder_model, validation_matrix)
    results_writer.write_all_information(auto_encoder_model, original, results, original_validation, results_validation)


if __name__ == "__main__":
    run_all_steps(auto_encoder, hyper_parameters)
    # run_deep_fully_connected_test()
    # run_all_steps(auto_encoder)
    # run_number_of_images_experiment()


#Next coding steps:
# Make a dev and test set to compare error: DONE
# Link frame extractor to the training engine (Parameterize this guy)
# Make a simple video to have an easy case: DONE
    #Investigate how you actutally want to do this.
# Program a shitty version of my idea
# Get U-net working: DONE
# Get U-net performing
# Figure out how to get the activated neron
# Parameterize the black and white better: DONE
# Figure out how to run this on the cluster that Saad told me about: DONE
# Make it easy to switch out videos: DONE
# Learn how to use tensorboard
# Look into learning rate decay better
# Make allow the framework to have a validation set
# Fix the bull shit on github: DONE
# Make a Fully convolutional network that isn't as huge as U-Net
# Set recipe configurations to make switching: DONE
# Make visualize show validation set too: DONE
# Make a method to use the model for visualizing without re-training
# Make the vectors into an objects so we know what the original images were trained on
# Save the training and validation sets along with the model so that when you try and retrain you don't get different training images
# Save the tensorboard models and the .h5 files into the results folders

# Make code pullable
# Git hub git ignore and choose python: DONE
# .env: NOT NECESSARY
# Select 10 videos and train against all them: ON its way
# Try hyperbolic tangent
# Try exponential
# Co lab

# Install tensorflow GPU: I don't have NVIDIA GPU's so this is not possible
# requirements / pip file to show what d
# ependencies to download: DONE
# Automate the video training to have a validation set: DONE
# Auto encoder non image data (not text) high dimention, didgets, mnist: DONE
# Get a cnn with mnist: DONE
# Fridays at noon
# pip freeze: DONE
# Get working with pre loaded datasets
# Make the image pulling more durable and useful to prepare for using many different datasets with little: DONE
# Change folder names and make sure it doesn't screw up pulling: DONE
# Save images to a file instead: DONE
# Mean absolute percentage error
# Write descriptions of your hyper-parameters so they make sense: DONE
# Normalize your images: Done

# make it run the python way
# rename research: DONE
# rename Training engine to main.py: DONE


# Write out plan
# Reread Unet
# Instance Segmentation
# Mask RCN unit
# Auto coloration
# Get working in color
# Get CNN working
# Sementement neuron








