import copy
import os
import shutil

import cv2

from Segmenter import StateObjectHelpers
from Segmenter.SingleImageObjectFinder import SingleImageObjectFinder
from Segmenter.StateObject import StateObject


class State:
    root_video_name = None
    current_frame_number = 0
    current_image = None
    is_object_thresh_hold = .001
    single_image_processor = None
    state_object_set = None
    state_object_set_history = None
    output_folder_template = "./OutputImages/object%d"

    def __init__(self, root_video_name):
        self.single_image_object_finder = SingleImageObjectFinder()
        self.state_object_set = []
        self.state_object_set_history = []
        self.root_video_name = root_video_name

    def update_state(self, segmented_image_for_update):
        self.current_image = segmented_image_for_update
        set_of_numbered_object_labels = self.single_image_processor.process_single_image(self.current_image)
        # Match objects with each other
        self.sort_state_objects_into_folders()

    def print_image_by_state_object_id(self, state_object, image):
        image_to_show = copy.copy(image)
        for i in range(image_to_show.shape[0]):
            for j in range(image_to_show.shape[1]):
                if image_to_show[i, j] != state_object.get_object_id():
                    image_to_show[i, j] = 0
            cv2.imwrite(state_object.get_folder_path() +
                        self.root_video_name + "/%d.png" % state_object.get_number_of_times_recorded(), image_to_show)
            state_object.advance_number_of_times_recorded()

    def sort_state_objects_into_folders(self, image):
        for state_object in self.state_object_set:
            state_object.set_folder_path(self.output_folder_template % state_object.get_object_id())
            self.create_or_delete_folder_if_necessary(state_object.get_folder_path())
            self.print_image_by_state_object_id(state_object, image)

    def create_or_delete_folder_if_necessary(self, filePath):
        if not os.path.isdir(filePath):
            os.makedirs(filePath)

    def map_single_image_output_to_state_objects(self, set_of_numbered_object_labels, segmented_image, set_of_current_state_objects):
        for single_image_number in set_of_numbered_object_labels:

    def create_new_set_of_state_objects_from_set_of_ids(self, set_of_numbered_object_labels, segmented_image):
        new_possible_objects = []
        for object_label in set_of_numbered_object_labels:
            current_object_to_add = StateObject(object_label)
            for i in range(segmented_image.shape[0]):
                for j in range(segmented_image.shape[1]):
                    if segmented_image[i, j] == object_label:
                        current_object_to_add.current_pixels.append((i,j))
            new_possible_objects.append(current_object_to_add)
        return new_possible_objects

    def find_centroids(self, list_of_state_objects):
        for state_objects in list_of_state_objects:
            state_objects.set_current_centroid(StateObjectHelpers.StateObjectHelpers.calculate_centroid(state_objects))


    #So I can think of a few senseable ways to do this. One intersection of pixels, centroid, and momentum
    #