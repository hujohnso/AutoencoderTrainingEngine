

from AutoEncoder.ImageManipulationType import ImageManipulationType


class ModelHyperParameters:
    def __init__(self):
        pass

    base_file_path = "../"
    file_path_for_frames = base_file_path + "FrameExtractor/Animations/"
    file_path_for_validation_set = base_file_path + ""
    file_path_for_training_set = base_file_path + ""
    number_of_epochs_for_training = 20
    batch_size = 30
    image_width_after_rescale: int
    image_height_after_rescale: int
    image_depth = 3

    as_gray = True
    original_video = "squareMovingTopRightToTopLeft"
    number_of_images = 120
    original_validation_video = "squareMovingTopLeftToBottomRight"
    number_of_images_for_validation = 120
    test_image_name = "Synfig Animation.0060.jpg"
    model_name = "my_model.h5"
    working_model_path = base_file_path + "saved_models/" + model_name
    tensor_board_directory = base_file_path + "tensor_board_models"
    load_model = False
    starting_frame_for_visualize = .0040

    # These parameters tell you what type of rescaling you would like to do
    # Note that if this variable is set to Pixel it will rescale according to pixels
    # Also note that if this variable is set to ratio it will rescale according to the ratio in image rescale value.
    type_of_transformation = ImageManipulationType.PIXEL
    image_rescale_value = 1.0 / 25
    pixel_resize_value = 64
    pixel_resize_for_visualize = 256
    adam_specify_learning_rate = False
    adam_alpha = None
    adam_decay_rate = None


class ModelHyperParametersRealImagesGray(ModelHyperParameters):
    def __init__(self):
        super().__init__()

    #This one has pretty good resutlts
    number_of_epochs_for_training = 10
    number_of_images = 120
    batch_size = 120
    starting_frame_for_visualize = .0060
    file_path_for_frames = ModelHyperParameters.base_file_path + "FrameExtractor/tmp/"
    original_video = "training_video"
    original_validation_video = "training_video"
    #converged down very well
    #model_name = "real_image_model_grey.h5"
    model_name = "real_image_model_grey_num_images_120.h5"
    load_model = False
    pixel_resize_value = 64
    adam_specify_learning_rate = True
    adam_alpha = 1e-4
    adam_decay_rate = .001
    file_path_for_validation_set = ModelHyperParameters.base_file_path + "FrameExtractor/tmp"
    file_path_for_training_set = ModelHyperParameters.base_file_path + "FrameExtractor/tmp"


class ModelHyperParametersRealImagesColor(ModelHyperParameters):
    def __init__(self):
        super().__init__()
    number_of_epochs_for_training = 1000
    number_of_images = 10
    batch_size = 10
    file_path_for_frames = ModelHyperParameters.base_file_path + "FrameExtractor/tmp/"
    original_video = "training_video"
    original_validation_video = "training_video"
    test_image_name = "training_video0.0080.jpg"
    model_name = "real_image_model_color.h5"
    load_model = False
    as_gray = False
    adam_specify_learning_rate = True
    adam_alpha = 1e-1
    adam_decay_rate = .0001
    file_path_for_validation_set = ModelHyperParameters.base_file_path + ""
    file_path_for_training_set = ModelHyperParameters.base_file_path + ""


class ModelHyperParametersAnimationGrey(ModelHyperParameters):
    def __init__(self):
        super().__init__()
    file_path_for_frames = ModelHyperParameters.base_file_path + "FrameExtractor/Animations/"
    number_of_epochs_for_training = 200
    batch_size = 30
    as_gray = True
    original_video = "squareMovingTopRightToTopLeft"
    number_of_images = 120
    original_validation_video = "squareMovingTopLeftToBottomRight"
    number_of_images_for_validation = 120
    test_image_name = "Synfig Animation.0060.jpg"
    model_name = "animation_grey.h5"
    load_model = False
    file_path_for_validation_set = ModelHyperParameters.base_file_path + ""
    file_path_for_training_set = ModelHyperParameters.base_file_path + ""
































