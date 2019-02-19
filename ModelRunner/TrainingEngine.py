import time

from AutoEncoder import FullyConnectedAutoEncoder
from ModelRunner.ModelHyperParameters import ModelHyperParametersRealImagesColor

# hyper_parameters = ModelHyperParameters()
#hyper_parameters = ModelHyperParametersRealImagesGray()
hyper_parameters = ModelHyperParametersRealImagesColor()
#hyper_parameters = ModelHyperParametersAnimationGrey()

auto_encoder = FullyConnectedAutoEncoder.FullyConnectedAutoEncoder(hyper_parameters)
# auto_encoder = ConvAutoEncoder.ConvAutoEncoder(hyper_parameters)
# auto_encoder = FullyConvolutionalAutoEncoder(hyper_parameters)


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


if __name__ == "__main__":
    run_all_steps(auto_encoder)

#Next coding steps:
# Make a dev and test set to compare error: No test but dev atleast :)
# Link frame extractor to the training engine (Parameterize this guy)
# Make a simple video to have an easy case
    #Investigate how you actutally want to do this.
# Program a shitty version of my idea
# Get U-net working: DONE
# Get U-net performing
# Figure out how to get the activated neron
# Parameterize the black and white better: DONE
# Figure out how to run this on the cluster that Saad told me about.
# Make it easy to switch out videos
# Learn how to use tensorboard
# Look into learning rate decay better
# Make a seperate video for testing the auto encoder
# Make allow the framework to test against that ^
# Fix the bull shit on github: DONE
# Make a Fully convolutional network that isn't as huge as U-Net
# Set recipe configurations to make switching
# Make visualize show validation set too
# Make a method to use the model for visualizing without re-training
# Make the vectors into an objects so we know what the original images were trained on


# Make code pullable
# Git hub git ignore and choose python
# .env
# Select 10 videos and train against all them
# Try hyperbolic tangent
# Try exponential
# Co lab

# Install tensorflow GPU
# requirements / pip file to show what dependencies to download
# Automate the video training to have a validation set
# Auto encoder non image data (not text) high dimention, didgets, mnist
# Get a cnn with mnist
# Fridays at noon
# pip freeze
# Get working with pre loaded datasets



#Before help
    #Set up hyperparmaters have different configurations to quickly switch out
    #Allow for switching switching


#Questions:
 #Does it make sense that static comes from relu activations?
 #Why does it struggle with more images and how do I combat this?
 #Can an autoEncoder match the same object in different places with out data agumentation.
 #What was the web site to run on the cluster for free
 #What are the important parts of tensorboard for me to know?
 #Is it correct to say that if my loss isn't converging then the network isn't deep enough?






