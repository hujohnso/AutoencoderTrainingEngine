import copy
import numpy

import cv2
import os


class StatePrinter:
    output_folder = "./Segmenter/Images/OutputImages"
    output_folder_template = output_folder + "/object%d/"

    def __init__(self):
        self.del_contents_of_output_folder()

    def del_contents_of_output_folder(self):
        for the_file in os.listdir(self.output_folder):
            file_path = os.path.join(self.output_folder, the_file)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
                # elif os.path.isdir(file_path): shutil.rmtree(file_path)
            except Exception as e:
                print(e)

    def print_image_by_state_object_id(self, state_object, image, original_image):
        image_filter = self.alter_image_for_element_wise_mult(state_object, image)
        image_to_show = self.filter_out_non_object_pixels(image_filter, original_image)
        cv2.imwrite(state_object.get_folder_path() + "%d.png" % state_object.get_number_of_times_recorded(),
                    image_to_show)
        state_object.advance_number_of_times_recorded()

    def filter_out_non_object_pixels(self, segmented_3_d, original_image):
        return numpy.multiply(segmented_3_d, original_image)

    def alter_image_for_element_wise_mult(self, state_object, image):
        image_to_show = copy.copy(image)
        for i in range(image_to_show.shape[0]):
            for j in range(image_to_show.shape[1]):
                if image_to_show[i, j] != state_object.current_mapped_number:
                    image_to_show[i, j] = 0
                else:
                    image_to_show[i, j] = 1
        return numpy.dstack([image_to_show] * 3)

    def sort_state_objects_into_folders(self, segmented_image, original_image, state_objects_to_print):
        for state_object in state_objects_to_print:
            state_object.set_folder_path(self.output_folder_template % state_object.get_object_id())
            self.create_or_delete_folder_if_necessary(state_object.get_folder_path())
            self.print_image_by_state_object_id(state_object, segmented_image, original_image)

    def create_or_delete_folder_if_necessary(self, file_path):
        if not os.path.isdir(file_path):
            os.makedirs(file_path)

    def combine_state_objects_into_video(self):
        for state_object_folder in os.listdir(self.output_folder):
            list_of_images = os.listdir(self.output_folder + "/" + state_object_folder)
            list_of_images.sort(key=lambda fname: int(fname.split('.')[0]))
            image_for_video_pixel_size = cv2.imread(self.output_folder + "/" + state_object_folder + "/" + list_of_images[1], 1)
            object_video = cv2.VideoWriter(self.output_folder + "/" + state_object_folder + "/" + "obj_video.avi",
                                           0,
                                           30,
                                           (image_for_video_pixel_size.shape[0], image_for_video_pixel_size.shape[1]))
            for frame in list_of_images:
                object_video.write(cv2.imread(self.output_folder + "/" + state_object_folder + "/" + "/" + frame, 1))
            cv2.destroyAllWindows()
            object_video.release()
