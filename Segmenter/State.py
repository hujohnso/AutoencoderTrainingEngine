import copy

import cv2


class State:
    current_image = None
    is_object_thresh_hold = .001

    def update_state(self, segmented_image_for_update):
        self.current_image = segmented_image_for_update

    def print_image_by_value(value, image, ):
        image_to_show = copy.copy(image)
        for i in range(image_to_show.shape[0]):
            for j in range(image_to_show.shape[1]):
                if image_to_show[i, j] != value:
                    image_to_show[i, j] = 0
            cv2.imwrite("./OutputImages/object%d" % value + ".png", image_to_show)

