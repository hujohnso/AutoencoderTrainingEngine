import time

from AutoEncoder import ConvAutoEncoder, FullyConnectedAutoEncoder
from ModelRunner.ModelHyperParameters import ModelHyperParameters

hyper_parameters = ModelHyperParameters()
# auto_encoder = FullyConnectedAutoEncoder.FullyConnectedAutoEncoder(hyper_parameters)
auto_encoder = ConvAutoEncoder.ConvAutoEncoder(hyper_parameters)


def timer(executable, function_executed):
    start_time = time.time()
    return_value = executable()
    end_time = time.time()
    print("It took ", end_time - start_time, " for ", function_executed, " to execute")
    return return_value


def run_all_steps(autoEncoder):
    input_matrix = timer(lambda: auto_encoder.init_training_matrix(), "image vector creation")
    auto_encoder_model = timer(lambda: auto_encoder.build_model(input_matrix), "model creation")
    auto_encoder_model = timer(lambda: auto_encoder.train(input_matrix, auto_encoder_model), "the model to train")
    timer(lambda: auto_encoder.visualize(auto_encoder_model), "visualize")


if __name__ == "__main__":
    run_all_steps(auto_encoder)

#Next coding steps: Make a dev an test set to compaire error
#Maybe set up a training strategy to set the batch size down
#Make a convolutional auto encoder
#Make U-net
#Attempt to train on more images.



