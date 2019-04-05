from Segmenter.ImageStreamCreator import ImageStreamCreator
from Segmenter.State import State


class SegmentLabelRunner:
    folder_with_frames = None

    def __init__(self, folder_with_frames, video_name):
        self.folder_with_frames = folder_with_frames
        self.video_name = video_name

    def run_segment_label_runner(self):
        image_stream_creator = ImageStreamCreator(self.folder_with_frames, self.video_name)
        image_stream = image_stream_creator.get_segmented_image_stream()
        state = State(self.video_name)
        for image in image_stream:
            state.update_state(image)

if __name__ == "__main__":
    segment_label_runner = SegmentLabelRunner("../AutoEncoderTrainer/FrameExtractor/Animations/easyPolygon/", "easyPolygon")
    segment_label_runner.run_segment_label_runner()

