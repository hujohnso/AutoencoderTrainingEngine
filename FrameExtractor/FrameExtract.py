from pytube import YouTube
import cv2
import shutil
import os


class FrameExtractor:
    # Squirrel: https://www.youtube.com/watch?v=6CCT1Crrd9Q
    # Elephant: https://www.youtube.com/watch?v=mlOiXMvMaZo
    # Golden Retriever: https://www.youtube.com/watch?v=C0taqxZ-gLQ
    # Turtle: https://www.youtube.com/watch?v=CsxSXVbNL0Q
    # sea cow: https://www.youtube.com/watch?v=k8n2FZU2VXY
    # orca: https://www.youtube.com/watch?v=n2vZBxf-8dc
    # monkey: https://www.youtube.com/watch?v=spMkaJp975s
    # horse: https://www.youtube.com/watch?v=xC2I7Xc3znQ
    # donky: https://www.youtube.com/watch?v=gROO7xSTxfY
    # goat: https://www.youtube.com/watch?v=nlYlNF30bVg

    list_of_videos = ["https://www.youtube.com/watch?v=r6y3moxAtXQ",
                      "https://www.youtube.com/watch?v=mlOiXMvMaZo",
                      "https://www.youtube.com/watch?v=C0taqxZ-gLQ",
                      "https://www.youtube.com/watch?v=CsxSXVbNL0Q",
                      "https://www.youtube.com/watch?v=k8n2FZU2VXY",
                      "https://www.youtube.com/watch?v=n2vZBxf-8dc",
                      "https://www.youtube.com/watch?v=spMkaJp975s",
                      "https://www.youtube.com/watch?v=xC2I7Xc3znQ",
                      "https://www.youtube.com/watch?v=gROO7xSTxfY",
                      "https://www.youtube.com/watch?v=nlYlNF30bVg"]

    number_of_images_from_each_video_train = 50

    def delete_folder_and_create_new_empty_folder(self, filePath):
        if os.path.isdir(filePath):
            os.chmod(filePath, 0o777)
            shutil.rmtree(filePath)
        os.makedirs(filePath)

    def download_and_save_video(self, file_path, video_url):
        video = YouTube(video_url)
        video = video.streams.first()
        file_path = file_path + "/"
        video.download(file_path)
        return video.default_filename

    def split_video_into_frames_and_save_pngs(self, video_name, parent_file_name, training_file_path, validation_file_path):
        vidcap = cv2.VideoCapture(parent_file_name + "/" + video_name)
        success, image = vidcap.read()
        count = 0
        frame_count = 1
        number_of_total_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
        number_of_frames_in_between = int(number_of_total_frames / (self.number_of_images_from_each_video_train * 1.2))
        while success:
            if frame_count % number_of_frames_in_between == 0:
                frame_name = "/" + video_name[:-4] + "%d.jpg" % count
                if count % 6 == 0:
                    cv2.imwrite(validation_file_path + frame_name, image)
                else:
                    cv2.imwrite(training_file_path + frame_name, image)
                count += 1
            success, image = vidcap.read()
            frame_count += 1

    def extract(self):
        parent_file_path = "./tmp"
        validation_file_path = parent_file_path + "/validation"
        train_file_path = parent_file_path + "/train"
        self.delete_folder_and_create_new_empty_folder(parent_file_path)
        os.makedirs(validation_file_path)
        os.makedirs(train_file_path)
        for video in self.list_of_videos:
            video_name = self.download_and_save_video(parent_file_path, video)
            self.split_video_into_frames_and_save_pngs(video_name,
                                                       parent_file_path,
                                                       train_file_path,
                                                       validation_file_path)


if __name__ == "__main__":
    frameExtractor = FrameExtractor()
    frameExtractor.extract()
