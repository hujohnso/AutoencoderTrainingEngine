from Segmenter.ImageStreamCreator import ImageStreamCreator
from Segmenter.State import State


class SegmentLabelRunner:
    folder_with_frames = None

    def __init__(self, folder_with_frames):
        self.folder_with_frames = folder_with_frames

    def run_segment_label_runner(self):
        image_stream_creator = ImageStreamCreator(self.folder_with_frames, "squareCircleStar")
        image_stream = image_stream_creator.get_segmented_image_stream()
        state = State("squareCircleStar")
        for image in image_stream:
            state.update_state(image)






if __name__ == "__main__":
    segment_label_runner = SegmentLabelRunner("../FrameExtractor/Animations/squareCircleStar/")
    segment_label_runner.run_segment_label_runner()

