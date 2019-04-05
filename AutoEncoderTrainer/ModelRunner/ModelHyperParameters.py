from AutoEncoderTrainer.AutoEncoder import ImageManipulationType


class ModelHyperParameters(object):
    def __init__(self):
        self.base_file_path = "../"
        self.file_path_for_frames = self.base_file_path + "FrameExtractor/Animations/"
        self.file_path_for_validation_set = self.base_file_path + ""
        self.file_path_for_training_set = self.base_file_path + ""
        self.number_of_epochs_for_training = 20
        self.batch_size = 30
        self.load_model = False
        self.adam_specify_learning_rate = False
        self.adam_alpha = 1e-3
        self.adam_decay_rate = .001
        self.as_gray = True
        self.number_of_images = 120
        self.number_of_images_for_validation = 120
        self.type_of_transformation = ImageManipulationType.PIXEL
        self.image_rescale_value = 1.0 / 25
        self.pixel_resize_value = 64
        self.pixel_resize_for_visualize = 256
        self.model_name = "my_model.h5"
        self.working_model_path = self.base_file_path + "saved_models/"
        self.tensor_board_directory = self.base_file_path + "tensor_board_models"

    # These parameters describe the necessary file paths to the sets of images to train on
    base_file_path = "../"
    file_path_for_frames = base_file_path + "FrameExtractor/Animations/"
    file_path_for_validation_set = base_file_path + ""
    file_path_for_training_set = base_file_path + ""
    results_folder = "tmp"

    # Training specific parameters
    number_of_epochs_for_training = 20
    batch_size = 30
    load_model = False
    adam_specify_learning_rate = False
    adam_alpha = None
    adam_decay_rate = None

    # Training and validation set definitions
    as_gray = True
    number_of_images = 120
    number_of_images_for_validation = 120
    type_of_transformation = ImageManipulationType.PIXEL
    image_rescale_value = 1.0 / 25
    pixel_resize_value = 64
    pixel_resize_for_visualize = 256

    # Model and Tensorboard definitions
    model_name = "my_model.h5"
    working_model_path = base_file_path + "saved_models/" + model_name
    tensor_board_directory = base_file_path + "tensor_board_models"


class ModelHyperParametersRealImagesGray(ModelHyperParameters):
    def __init__(self):
        super().__init__()
        self.number_of_epochs_for_training = 15
        self.number_of_images = 50
        self.number_of_images_for_validation = 5
        self.batch_size = 5
        self.load_model = True
        self.pixel_resize_value = 64
        self.adam_specify_learning_rate = True
        self.adam_alpha = 0.000001
        self.adam_decay_rate = None
        self.results_folder = "print_in_the_correct_location"
        self.model_name = self.results_folder + ".h5"
        self.working_model_path = self.base_file_path + "Results/" + self.results_folder + "/"
        self.file_path_for_validation_set = ModelHyperParameters.base_file_path + "FrameExtractor/tmp/validation"
        self.file_path_for_training_set = ModelHyperParameters.base_file_path + "FrameExtractor/tmp/train"
        self.tensor_board_directory = ModelHyperParameters.base_file_path + "Results/" + self.results_folder + "/"


class ModelHyperParametersRealImagesColor(ModelHyperParameters):
    def __init__(self):
        super().__init__()
        self.number_of_epochs_for_training = 2
        self.number_of_images = 300
        self.number_of_images_for_validation = 50
        self.batch_size = 32
        self.load_model = False
        self.as_gray = False
        self.adam_specify_learning_rate = True
        self.adam_alpha = 0.00001
        self.adam_decay_rate = None
        self.pixel_resize_value = 64
        self.results_folder = "terrible_2"
        self.model_name = self.results_folder + ".h5"
        self.working_model_path = self.base_file_path + "Results/" + self.results_folder + "/"
        self.file_path_for_validation_set = ModelHyperParameters.base_file_path + "FrameExtractor/tmp/validation"
        self.file_path_for_training_set = ModelHyperParameters.base_file_path + "FrameExtractor/tmp/train"
        self.tensor_board_directory = ModelHyperParameters.base_file_path + "Results/" + self.results_folder + "/"


class ModelHyperParametersAnimationGrey(ModelHyperParameters):
    def __init__(self):
        super().__init__()
        self.number_of_epochs_for_training = 200
        self.batch_size = 30
        self.as_gray = True
        self.number_of_images = 120
        self.number_of_images_for_validation = 120
        self.model_name = "animation_grey.h5"
        self.load_model = False
        self.file_path_for_validation_set = ModelHyperParameters.base_file_path + ""
        self.file_path_for_training_set = ModelHyperParameters.base_file_path + ""


class ModelHyperParametersMNIST(ModelHyperParameters):
    def __init__(self):
        super().__init__()
        self.number_of_epochs_for_training = 1500
        self.batch_size = 256
        self.as_gray = True
        self.number_of_images = 5016
        self.number_of_images_for_validation = 256
        self.model_name = "mnist_2048.h5"
        self.load_model = False
        self.file_path_for_validation_set = ModelHyperParameters.base_file_path + "PreMadeDatasets/MNIST/mnist_jpgfiles/mnist_jpgfiles/train"
        self.file_path_for_training_set = ModelHyperParameters.base_file_path + "PreMadeDatasets/MNIST/mnist_jpgfiles/mnist_jpgfiles/test"
        self.type_of_transformation = ImageManipulationType.PIXEL
        self.image_rescale_value = 1.0 / 25
        self.pixel_resize_value = 64
        self.pixel_resize_for_visualize = 64
        self.results_folder = "mnist_2048_CONV_NET"
        self.adam_specify_learning_rate = True
        self.adam_alpha = 0.00001
        self.adam_decay_rate = None



