from Segmenter.ImageStreamCreator import ImageStreamCreator
from Segmenter.StateAdvancer import StateAdvancer
from Segmenter.model.State import State


class SegmentLabelRunner:
    folder_with_frames = None

    def __init__(self, folder_with_frames, video_name):
        self.folder_with_frames = folder_with_frames
        self.video_name = video_name
        self.image_stream_creator = ImageStreamCreator(self.folder_with_frames, self.video_name)

    def run_segment_label_runner(self):
        original_image_stream = self.image_stream_creator.get_original_images()
        segmented_image_stream = self.image_stream_creator.get_segmented_image_stream()
        state_advancer = StateAdvancer(self.video_name, original_image_stream, segmented_image_stream)
        state_advancer.advance_state_through_all_frames()

    def get_segment_labels(self):
        return self.image_stream_creator.get_state_object_classes_for_training()


