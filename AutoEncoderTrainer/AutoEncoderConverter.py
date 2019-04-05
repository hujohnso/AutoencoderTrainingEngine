# This class will be responsible for ripping out the decoding layers, and putting on and training classification layers
from keras import Model
from keras.layers import Dense
from keras.optimizers import Adadelta

from AutoEncoderTrainer.AutoEncoder.AutoEncoderDefinitions.ConvAutoEncoder import ConvAutoEncoder
from AutoEncoderTrainer.AutoEncoderTrainer import AutoEncoderTrainer
from AutoEncoderTrainer.ModelRunner.ModelHyperParameters import ModelHyperParametersSimpleAnimationColor


class AutoEncoderConverter:
    def __init__(self, trained_auto_encoder, number_of_layers_to_pop):
        self.trained_auto_encoder = trained_auto_encoder
        self.number_of_layers_to_pop = number_of_layers_to_pop

    def convert_autoencoder_into_object_classifier(self):
        print(self.trained_auto_encoder.summary())
        self.remove_decoding_layers()
        self.freeze_pretrained_layers_weights()
        self.add_classification_layers_to_model(3)
        print(self.trained_auto_encoder.summary)

    def freeze_pretrained_layers_weights(self):
        for layer in self.trained_auto_encoder.layers:
            layer.trainable = False
        self.trained_auto_encoder = Model(inputs=self.trained_auto_encoder.inputs,
                                          outputs=self.trained_auto_encoder.output)

    def remove_decoding_layers(self):
        for i in range(self.number_of_layers_to_pop):
            self.pop_layer(self.trained_auto_encoder)

    def add_classification_layers_to_model(self, num_classes):
        x = Dense(num_classes, activation='softmax')(self.trained_auto_encoder.output)
        model = Model(inputs=self.trained_auto_encoder.inputs, outputs=x)
        model.compile(optimizer=Adadelta(), loss='categorical_crossentropy', metrics=['accuracy'])
        self.trained_auto_encoder = model

    def pop_layer(self, model):
        if not model.outputs:
            raise Exception('Sequential model cannot be popped: model is empty.')
        model.layers.pop()
        if not model.layers:
            model.outputs = []
            model.inbound_nodes = []
            model.outbound_nodes = []
        else:
            model.layers[-1].outbound_nodes = []
            model.outputs = [model.layers[-1].output]
        model.built = False


if __name__ == "__main__":
    params = ModelHyperParametersSimpleAnimationColor()
    auto_encoder = ConvAutoEncoder(params)
    auto_encoder_trainer = AutoEncoderTrainer()
    trained_model = auto_encoder_trainer.run_all_steps(auto_encoder, params)
    auto_encoder_converter = AutoEncoderConverter(trained_model, 1)


