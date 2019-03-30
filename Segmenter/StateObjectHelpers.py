# This file is for a bunch of static helper functions so that a few classes don't get crowded with functions that
# don't need to be confined to that class
import math

from Segmenter.Centroid import Centroid


class StateObjectHelpers:
    @staticmethod
    def calculate_centroid(state_object):
        x_total = 0
        y_total = 0
        num_pixels = len(state_object.current_pixels)
        for i in range(num_pixels):
            x_total += state_object.current_pixels[i, 0]
            y_total += state_object.current_pixels[i, 1]
        return Centroid(x_total / num_pixels,
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