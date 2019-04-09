import numpy

from Segmenter.StatePrinter import StatePrinter
from Segmenter.SingleImageObjectFinder import SingleImageObjectFinder
from Segmenter.Utils import StateObjectHelpers
from Segmenter.model.State import State
from Segmenter.model.StateObject import StateObject


class StateAdvancer:

    def __init__(self, root_video_name, original_images, segmented_images):
        self.state = State(root_video_name)
        self.segmented_images = segmented_images
        self.state_printer = StatePrinter()
        self.single_image_object_finder = SingleImageObjectFinder()
        self.current_image = None
        self.original_images = original_images

    def advance_state_through_all_frames(self):
        if self.original_images.shape[0] != self.segmented_images.shape[0]:
            raise Exception("You have a different number of original images and segmented images... Something is up with the image stream creator")
        for i in range(len(self.segmented_images)):
            self.update_state(self.segmented_images[i], self.original_images[i])

    def initialize_state(self, initial_segmented_image, initial_original_image):
        self.current_image = initial_segmented_image
        list_of_numbered_object_labels = self.single_image_object_finder.process_single_image(self.current_image)
        self.state.state_object_set = self.create_new_set_of_state_objects_from_set_of_ids(
            list_of_numbered_object_labels,
            self.current_image)
        self.find_centroids(self.state.state_object_set)
        self.state_printer.sort_state_objects_into_folders(initial_segmented_image, initial_original_image, self.state.state_object_set)

    def update_state(self, segmented_image_for_update, original_image):
        if self.state.current_frame_number == 0:
            self.initialize_state(segmented_image_for_update, original_image)
        else:
            self.current_image = segmented_image_for_update
            set_of_numbered_object_labels = self.single_image_object_finder.process_single_image(self.current_image)
            self.state.state_object_set = self.map_single_image_output_to_state_objects(set_of_numbered_object_labels,
                                                                                        self.current_image,
                                                                                        self.state.state_object_set)
            self.state_printer.sort_state_objects_into_folders(self.current_image, original_image, self.state.state_object_set)
        self.state.current_frame_number += 1

    def map_single_image_output_to_state_objects(self, set_of_numbered_object_labels, segmented_image,
                                                 set_of_current_state_objects):
        new_state_objects = self.create_new_set_of_state_objects_from_set_of_ids(set_of_numbered_object_labels,
                                                                                 segmented_image)
        new_state_objects = self.find_centroids(new_state_objects)
        probability_matrix = self.create_and_fill_probability_matrix(new_state_objects, set_of_current_state_objects)
        chosen_indices = self.traverse_the_probability_matrix(probability_matrix)
        return self.map_new_state_object_indices_to_current_state_objects(chosen_indices, new_state_objects,
                                                                          set_of_current_state_objects)

        # For now just assume all objects that are on the screen will stay and none will be introduced
        # Also assume that none run into each other
        # also none are compsite objects

    def create_and_fill_probability_matrix(self, set_of_new_state_objects, set_of_current_state_objects):
        if len(set_of_new_state_objects) != len(set_of_current_state_objects):
            print("One object disappeared :(")
        probability_matrix = numpy.zeros((len(set_of_new_state_objects), len(set_of_current_state_objects)))
        for new_state_object_index in range(len(set_of_new_state_objects)):
            for current_state_object_index in range(len(set_of_current_state_objects)):
                probability_matrix[new_state_object_index, current_state_object_index] = \
                    StateObjectHelpers.StateObjectHelpers.get_centroid_distance(
                        set_of_new_state_objects[new_state_object_index].current_centroid,
                        set_of_current_state_objects[current_state_object_index].current_centroid)
        return probability_matrix

        # There could be a ton of ways to do this however for now I am just doing the simplest thing which is to choose the greedy way for now

    def traverse_the_probability_matrix(self, probability_matrix):
        chosen_indices = []
        for new_state_object_index in range(probability_matrix.shape[0]):
            minimum_centroid_distance = 244 * 244
            minimum_centroid_index = 0
            for current_state_object_index in range(probability_matrix.shape[1]):
                if probability_matrix[new_state_object_index, current_state_object_index] < minimum_centroid_distance:
                    minimum_centroid_index = current_state_object_index
                    if minimum_centroid_index not in chosen_indices:
                        minimum_centroid_distance = probability_matrix[
                            new_state_object_index, current_state_object_index]

            chosen_indices.append(minimum_centroid_index)
        return chosen_indices

    def map_new_state_object_indices_to_current_state_objects(self, chosen_indices, list_of_new_state_objects,
                                                              list_of_current_state_objects):
        for i in range(len(chosen_indices)):
            list_of_current_state_objects[chosen_indices[i]].current_mapped_number = list_of_new_state_objects[
                i].object_id
        return list_of_current_state_objects

    def create_new_set_of_state_objects_from_set_of_ids(self, set_of_numbered_object_labels, segmented_image):
        new_possible_objects = []
        for object_label in set_of_numbered_object_labels:
            new_possible_objects.append(StateObject(object_label))
            for i in range(segmented_image.shape[0]):
                for j in range(segmented_image.shape[1]):
                    if segmented_image[i, j] == object_label:
                        new_possible_objects[-1].current_pixels.append((i, j))
        return new_possible_objects

    def find_centroids(self, list_of_state_objects):
        for state_objects in list_of_state_objects:
            state_objects.set_current_centroid(StateObjectHelpers.StateObjectHelpers.calculate_centroid(state_objects))
        return list_of_state_objects