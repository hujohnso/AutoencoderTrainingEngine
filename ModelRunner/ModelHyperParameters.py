class ModelHyperParameters:
    def __init__(self):
        pass
    base_file_path = "/home/hujohnso/Documents/Research2018/"
    file_path_for_frames = base_file_path + "FrameExtractor/tmp/"
    number_of_images = 300
    number_of_epochs_for_training = 5
    batch_size = 300
    image_width_after_rescale: int
    image_height_after_rescale: int
    image_depth = 3
    image_rescale_value = 1.0 / 15.0
    as_gray = True
    original_video = "Cute Baby Holland Lop Bunnies Playing Inside the House.mp4"
    test_image_name = "Cute Baby Holland Lop Bunnies Playing Inside the House.mp4frame400.jpg"
    model_name = "my_model.h5"
    working_model_path = base_file_path + "saved_models/" + model_name
    load_model = True
