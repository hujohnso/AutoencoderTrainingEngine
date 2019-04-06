from Segmenter.ImageStreamCreator import ImageStreamCreator
from Segmenter.StateAdvancer import StateAdvancer
from Segmenter.model.State import State


class SegmentLabelRunner:
    folder_with_frames = None

    def __init__(self, folder_with_frames, video_name):
        self.folder_with_frames = folder_with_frames
        self.video_name = video_name

    def run_segment_label_runner(self):
        image_stream_creator = ImageStreamCreator(self.folder_with_frames, self.video_name)
        original_image_stream = image_stream_creator.get_original_images()
        segmented_image_stream = image_stream_creator.get_segmented_image_stream()
        state_advancer = StateAdvancer(self.video_name, original_image_stream, segmented_image_stream)
        state_advancer.advance_state_through_all_frames()

if __name__ == "__main__":
    segment_label_runner = SegmentLabelRunner("../AutoEncoderTrainer/FrameExtractor/Animations/easyPolygon/", "easyPolygon")
    segment_label_runner.run_segment_label_runner()

