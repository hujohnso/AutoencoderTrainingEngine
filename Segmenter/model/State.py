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
    is_object_thresh_hold = .001
    single_image_object_finder = None
    state_object_set = None
    state_object_set_history = None

    def __init__(self, root_video_name):
        self.state_object_set = []
        self.state_object_set_history = []
        self.root_video_name = root_video_name
