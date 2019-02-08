from AutoEncoder.ImageManipulationType import ImageManipulationType


class ModelHyperParameters:
    def __init__(self):
        pass

    base_file_path = "/home/hujohnso/Documents/Research2018/"
    file_path_for_frames = base_file_path + "FrameExtractor/Animations/"
    number_of_epochs_for_training = 100
    batch_size = 30
    image_width_after_rescale: int
    image_height_after_rescale: int
    image_depth = 3

    as_gray = True

    # original_video = "Cute Baby Holland Lop Bunnies Playing Inside the House.mp4"
    # test_image_name = "Cute Baby Holland Lop Bunnies Playing Inside the House.mp4frame400.jpg"
    original_video = "Synfig Animation"
    number_of_images = 120
    original_validation_video = "Synfig Animation"
    number_of_images_for_validation = 120
    # test_image_name = "Cute Baby Holland Lop Bunnies Playing Inside the House.mp4frame400.jpg"
    test_image_name = "Synfig Animation.0060.jpg"
    model_name = "my_model.h5"
    working_model_path = base_file_path + "saved_models/" + model_name
    tensor_board_directory = base_file_path + "tensor_board_models"
    load_model = False
    frame_skipping_factor = 1

    # These parameters tell you what type of rescaling you would like to do
    # Note that if this variable is set to Pixel it will rescale according to pixels
    # Also note that if this variable is set to ratio it will rescale according to the ratio in image rescale value.
    type_of_transformation = ImageManipulationType.PIXEL
    image_rescale_value = 1.0 / 25
    pixel_resize_value = 128
