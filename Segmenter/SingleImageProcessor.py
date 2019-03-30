import numpy
import cv2
from Segmenter.UnetLoader import UnetLoader


class SingleImageProcessor:
    set_of_objects = set()
    object_index = 20
    new_image = None
    segmenter = None
    is_object_thresh_hold = .001
    # The object must be 1/100 th of the entire frame to be considered an object

    def __init__(self):
        u_net_loader = UnetLoader()
        self.segmenter = u_net_loader.load_unet()

    def check_if_pixel_tagged_or_not_one(self, row, column, img):
        if img[row, column] == 1 and not self.set_of_objects.issuperset({int(img[row, column])}):
            return True
        else:
            return False

    def perform_function_if_indicies_are_in_range(self, execute, row, column, img):
        if 0 <= row < img.shape[0] and 0 <= column < img.shape[1]:
            execute(row, column, img)

    def cluster_and_label_a_group_of_pixels(self, root_i, root_j, image):
        if self.check_if_pixel_tagged_or_not_one(root_i, root_j, image):
            image[root_i, root_j] = self.object_index
            self.perform_function_if_indicies_are_in_range(
                lambda x, y, z: self.cluster_and_label_a_group_of_pixels(x, y, z),
                root_i - 1, root_j, image)
            self.perform_function_if_indicies_are_in_range(
                lambda x, y, z: self.cluster_and_label_a_group_of_pixels(x, y, z),
                root_i - 1, root_j - 1, image)
            self.perform_function_if_indicies_are_in_range(
                lambda x, y, z: self.cluster_and_label_a_group_of_pixels(x, y, z),
                root_i - 1, root_j + 1, image)
            self.perform_function_if_indicies_are_in_range(
                lambda x, y, z: self.cluster_and_label_a_group_of_pixels(x, y, z),
                root_i, root_j + 1, image)
            self.perform_function_if_indicies_are_in_range(
                lambda x, y, z: self.cluster_and_label_a_group_of_pixels(x, y, z),
                root_i, root_j - 1, image)
            self.perform_function_if_indicies_are_in_range(
                lambda x, y, z: self.cluster_and_label_a_group_of_pixels(x, y, z),
                root_i + 1, root_j, image)
            self.perform_function_if_indicies_are_in_range(
                lambda x, y, z: self.cluster_and_label_a_group_of_pixels(x, y, z),
                root_i + 1, root_j - 1, image)
            self.perform_function_if_indicies_are_in_range(
                lambda x, y, z: self.cluster_and_label_a_group_of_pixels(x, y, z),
                root_i + 1, root_j + 1, image)

    def process_single_image_by_file_name(self, image_file_name):
        image = cv2.imread(image_file_name)
        image = cv2.resize(image, (224, 224))
        self.process_single_image(image)

    def process_single_image(self, image_to_process):
        seg_image = self.segmenter.predict(image_to_process.reshape(1, 224, 224, 3)).reshape(224, 224, 1)
        self.label_image(seg_image)
        self.filter_object_noise(seg_image)

    def label_image(self, image_to_process):
        for i in range(224):
            for j in range(224):
                if self.check_if_pixel_tagged_or_not_one(i, j, image_to_process):
                    self.set_of_objects.add(self.object_index)
                    self.cluster_and_label_a_group_of_pixels(i, j, image_to_process)
                    self.object_index += 10

    def close_to(self, float_val, desired_val, tol):
        value_to_check = float_val / desired_val
        if (1 - tol) < value_to_check < (1 + tol):
            return True
        else:
            return False

    def filter_object_noise(self, image):
        pixels_in_object = numpy.zeros([image.shape[0], image.shape[1]])
        number_of_pixels_in_object = 0
        objects_to_remove = set()
        for object_index in self.set_of_objects:
            for i in range(image.shape[0]):
                for j in range(image.shape[1]):
                    if image[i, j] == object_index:
                        number_of_pixels_in_object += 1
                        pixels_in_object[i, j] = 1
            if pixels_in_object < self.is_object_thresh_hold * image.shape[0] * image.shape[1]:
                objects_to_remove.add(object_index)
                for i in range(image.shape[0]):
                    for j in range(image.shape[1]):
                        if pixels_in_object[i, j] == 1:
                            image[i, j] = 0
        self.set_of_objects.remove(objects_to_remove)
