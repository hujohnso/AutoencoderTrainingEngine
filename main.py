import time

from AutoEncoder import FullyConnectedAutoEncoder
from AutoEncoder.ConvFullyConnectedUpConv import ConvFullyConnectedUpConv
from ModelRunner.ModelHyperParameters import ModelHyperParametersRealImagesColor, ModelHyperParametersRealImagesGray

# hyper_parameters = ModelHyperParameters()
from Results import ResultsWriter

hyper_parameters = ModelHyperParametersRealImagesGray()

# hyper_parameters = ModelHyperParametersRealImagesColor()
#hyper_parameters = ModelHyperParametersAnimationGrey()
# auto_encoder = ConvFullyConnectedUpConv(hyper_parameters)
auto_encoder = FullyConnectedAutoEncoder.FullyConnectedAutoEncoder(hyper_parameters)
# auto_encoder = ConvAutoEncoder.ConvAutoEncoder(hyper_parameters)
# auto_encoder = Unet()


def timer(executable, function_executed):
    start_time = time.time()
    return_value = executable()
    end_time = time.time()
    print("It took ", end_time - start_time, " for ", function_executed, " to execute")
    return return_value


def run_all_steps(autoEncoder):
    input_matrix = timer(lambda: auto_encoder.init_training_matrix(), "training set creation")
    validation_matrix = timer(lambda: autoEncoder.init_validation_matrix(), "validation/dev set creation")
    auto_encoder_model = timer(lambda: auto_encoder.build_model(input_matrix), "model creation")
    auto_encoder_model = timer(lambda: auto_encoder.train(input_matrix, auto_encoder_model, validation_matrix), "the model to train")
    timer(lambda: auto_encoder.visualize(auto_encoder_model), "visualize")
    results_writer = ResultsWriter.ResultsWriter(hyper_parameters)
    results_writer.write_results_to_file()



if __name__ == "__main__":
    run_all_steps(auto_encoder)

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
# Make it easy to switch out videos
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

# Make code pullable
# Git hub git ignore and choose python: DONE
# .env: NOT NECESSARY
# Select 10 videos and train against all them: ON its way
# Try hyperbolic tangent
# Try exponential
# Co lab

# Install tensorflow GPU
# requirements / pip file to show what dependencies to download: DONE
# Automate the video training to have a validation set: DONE
# Auto encoder non image data (not text) high dimention, didgets, mnist
# Get a cnn with mnist
# Fridays at noon
# pip freeze: DONE
# Get working with pre loaded datasets
# Make the image pulling more durable and useful to prepare for using many different datasets with little: DONE
# Change folder names and make sure it doesn't screw up pulling
# write some sort of
# Save images to a file instead
# Mean absolute percentage error
# Write descriptions of your hyper-parameters so they make sense
# Normalize your images

# make it run the python way
# rename research
# rename Training engine to main.py: DONE



#Before help
    #Set up hyperparmaters have different configurations to quickly switch out


#Questions:
    #Does the exploding gradients/ near zero gradients which leads residual networks affect gradient updates







