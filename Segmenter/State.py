import copy
import numpy
import os

import cv2

from Segmenter.Utils import StateObjectHelpers
from Segmenter.SingleImageObjectFinder import SingleImageObjectFinder
from Segmenter.model.StateObject import StateObject


# This class is too long
# State should be in Model. there needs to be two classes associated with this

class State:
    root_video_name = None
    current_frame_number = 0
    current_image = None
    is_object_thresh_hold = .001
    single_image_object_finder = None
    state_object_set = None
    state_object_set_history = None
    output_folder_template = "../Segmenter/Images/OutputImages/object%d/"

    def __init__(self, root_video_name):
        self.single_image_object_finder = SingleImageObjectFinder()
        self.state_object_set = []
        self.state_object_set_history = []
        self.root_video_name = root_video_name

    def initialize_state(self, initial_segmented_image):
        self.current_image = initial_segmented_image
        list_of_numbered_object_labels = self.single_image_object_finder.process_single_image(self.current_image)
        self.state_object_set = self.create_new_set_of_state_objects_from_set_of_ids(list_of_numbered_object_labels,
                                                                                     self.current_image)
        self.sort_state_objects_into_folders(initial_segmented_image)

    def update_state(self, segmented_image_for_update):
        if self.current_frame_number == 0:
            self.initialize_state(segmented_image_for_update)
        else:
            self.current_image = segmented_image_for_update
            set_of_numbered_object_labels = self.single_image_object_finder.process_single_image(self.current_image)
            self.state_object_set = self.map_single_image_output_to_state_objects(set_of_numbered_object_labels,
                                                                                  self.current_image,
                                                                                  self.state_object_set)
            self.sort_state_objects_into_folders(self.current_image)
        self.current_frame_number += 0


    def print_image_by_state_object_id(self, state_object, image):
        image_to_show = copy.copy(image)
        for i in range(image_to_show.shape[0]):
            for j in range(image_to_show.shape[1]):
                if image_to_show[i, j] != state_object.current_mapped_number:
                    image_to_show[i, j] = 0
                else:
                    image_to_show[i, j] = 20
        cv2.imwrite(state_object.get_folder_path() + "%d.png" % state_object.get_number_of_times_recorded(), image_to_show)
        state_object.advance_number_of_times_recorded()

    def sort_state_objects_into_folders(self, image):
        for state_object in self.state_object_set:
            state_object.set_folder_path(self.output_folder_template % state_object.get_object_id())
            self.create_or_delete_folder_if_necessary(state_object.get_folder_path())
            self.print_image_by_state_object_id(state_object, image)

    def create_or_delete_folder_if_necessary(self, filePath):
        if not os.path.isdir(filePath):
            os.makedirs(filePath)

    def map_single_image_output_to_state_objects(self, set_of_numbered_object_labels, segmented_image,
                                                 set_of_current_state_objects):
        new_state_objects = self.create_new_set_of_state_objects_from_set_of_ids(set_of_numbered_object_labels,
                                                                                 segmented_image)
        new_state_objects = self.find_centroids(new_state_objects)
        probability_matrix = self.create_and_fill_probability_matrix(new_state_objects, set_of_current_state_objects)
        chosen_indicies = self.traverse_the_probability_matrix(probability_matrix)
        return self.map_new_state_object_indices_to_current_state_objects(chosen_indicies, new_state_objects,
                                                                          set_of_current_state_objects)

    # For now just assume all objects that are on the screen will stay and none will be introduced
    # Also assume that none run into each other
    # also none are compsite objects
    def create_and_fill_probability_matrix(self, set_of_new_state_objects, set_of_current_state_objects):
        if len(set_of_new_state_objects) != len(set_of_current_state_objects):
            print("One object disappeared :(")
        probability_matrix = numpy.zeros(len(set_of_new_state_objects), len(set_of_current_state_objects))
        for new_state_object_index in range(len(set_of_new_state_objects)):
            for current_state_object_index in range(len(set_of_current_state_objects)):
                probability_matrix[new_state_object_index, current_state_object_index] = \
                    StateObjectHelpers.StateObjectHelpers.get_centroid_distance(
                        set_of_new_state_objects[new_state_object_index],
                        set_of_current_state_objects[current_state_object_index])
        return probability_matrix

    # There could be a ton of ways to do this however for now I am just doing the simplest thing which is to choose the greedy way for now
    def traverse_the_probability_matrix(self, probability_matrix):
        chosen_indices = []
        for new_state_object_index in range(probability_matrix.shape[0]):
            minimum_centroid_distance = None
            minimum_centroid_index: float
            for current_state_object_index in range(probability_matrix.shape[1]):
                if minimum_centroid_distance is not None and probability_matrix[
                    new_state_object_index, current_state_object_index] < minimum_centroid_distance:
                    if minimum_centroid_index not in chosen_indices:
                        minimum_centroid_index = current_state_object_index
                        minimum_centroid_distance = probability_matrix[
                            new_state_object_index, current_state_object_index]
            chosen_indices.append(minimum_centroid_index)
        return chosen_indices

    def map_new_state_object_indices_to_current_state_objects(self, chosen_indices, list_of_new_state_objects,
                                                              list_of_current_state_objects):
        for i in range(len(chosen_indices)):
            list_of_current_state_objects[chosen_indices[i]].current_mapped_number = list_of_new_state_objects[
                i].objectId
        return list_of_current_state_objects

    def create_new_set_of_state_objects_from_set_of_ids(self, set_of_numbered_object_labels, segmented_image):
        new_possible_objects = []
        for object_label in set_of_numbered_object_labels:
            current_object_to_add = StateObject(object_label)
            for i in range(segmented_image.shape[0]):
                for j in range(segmented_image.shape[1]):
                    if segmented_image[i, j] == object_label:
                        current_object_to_add.current_pixels.append((i, j))
            new_possible_objects.append(current_object_to_add)
        return new_possible_objects

    def find_centroids(self, list_of_state_objects):
        for state_objects in list_of_state_objects:
            state_objects.set_current_centroid(StateObjectHelpers.StateObjectHelpers.calculate_centroid(state_objects))
        return list_of_state_objects

    # So I can think of a few senseable ways to do this. One intersection of pixels, centroid, and momentum
    #
