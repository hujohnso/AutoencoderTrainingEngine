import numpy

import cv2

from Segmenter.UnetLoader import UnetLoader


class ImageStreamCreator:
    folder_with_images = None
    u_net_required_dim = 224

    def __init__(self, folder_with_images):
        self.folder_with_images = folder_with_images

    # This logic is simular to the logic in Frame AutoEncoder.  Consider making a util for this
    def get_segmented_image_stream(self):
        matrix = None
        num_in_matrix = 0
        file_names = cv2.os.listdir(self.folder_with_images)
        file_names.sort()
        for filename in file_names:
            if not filename.endswith(".png"):
                continue
            image = cv2.imread(self.folder_with_images + "/" + filename, 1)
            image = cv2.resize(image, (self.u_net_required_dim, self.u_net_required_dim))
            if num_in_matrix == 0:
                matrix = numpy.empty([len(cv2.os.listdir(self.folder_with_images)),
                                      self.u_net_required_dim,
                                      self.u_net_required_dim, 3])
            matrix[num_in_matrix, :, :, :] = image
            num_in_matrix += 1
        return self.segment_image_stream(matrix)

    def segment_image_stream(self, original_image_matrix):
        u_net_loader = UnetLoader()
        u_net = u_net_loader.load_unet()
        return u_net.predict(original_image_matrix)


