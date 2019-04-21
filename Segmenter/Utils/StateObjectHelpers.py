# This file is for a bunch of static helper functions so that a few classes don't get crowded with functions that
# don't need to be confined to that class
import math

from Segmenter.model.PointTuple import PointTuple


class StateObjectHelpers:
    @staticmethod
    def calculate_centroid(state_object):
        x_total = 0
        y_total = 0
        num_pixels = len(state_object.current_pixels)
        for i in range(num_pixels):
            x_total += state_object.current_pixels[i][0]
            y_total += state_object.current_pixels[i][1]
        return PointTuple(x_total / num_pixels,
                          y_total / num_pixels)

    @staticmethod
    def calculate_overlap(pixels_for_state_object_one, pixels_for_state_object_two):
        # Note this could be super expensive so maybe wait on this one
        es = "s"

    @staticmethod
    def get_centroid_distance(centroid_one, centroid_two):
        x_dist = abs(centroid_one.x - centroid_two.x)
        y_dist = abs(centroid_one.y - centroid_two.y)
        return math.sqrt(math.pow(x_dist, 2) + math.pow(y_dist, 2))

    @staticmethod
    def get_difference_in_mass(state_object_one, state_object_two):
        return len(state_object_two.current_pixels) - len(state_object_one.current_pixels)

    @staticmethod
    def get_difference_in_momentum(state_object_one, state_object_two):
        if state_object_one.last_centroid is None:
            return 0
        old_momentum = StateObjectHelpers.calculate_x_y_distance(state_object_one.current_centroid, state_object_one.last_centroid)
        current_momentum = StateObjectHelpers.calculate_x_y_distance(state_object_one.current_centroid, state_object_two.current_centroid)
        return StateObjectHelpers.get_centroid_distance(old_momentum, current_momentum)

    @staticmethod
    def calculate_x_y_distance(point_tuple_one, point_tuple_two):
            return PointTuple(point_tuple_one.x - point_tuple_two.x, point_tuple_one.y - point_tuple_two.y)
