from pytube import YouTube
import cv2
import shutil
import os


def delete_folder_and_create_new_empty_folder(filePath):
    if os.path.isdir(filePath):
        os.chmod(filePath, 0o777)
        shutil.rmtree(filePath)
    os.makedirs(filePath)


def download_and_save_video(filePath):
    video = YouTube("https://www.youtube.com/watch?v=aGE39IsemFw")
    video = video.streams.first()
    filePath = filePath + "/"
    video.download(filePath)
    return filePath + video.default_filename


def split_video_into_frames_and_save_pngs(video_file_path, filePath):
    vidcap = cv2.VideoCapture(video_file_path)
    success, image = vidcap.read()
    count = 0
    frame_count = 1
    number_of_frames_in_between = 1
    while success:
        if frame_count % number_of_frames_in_between == 0:
            cv2.imwrite(filePath + "training_video%.4f.jpg" % (.0001 * count), image)
            count += 1
        success, image = vidcap.read()
        print('Read a new frame: ', success)
        frame_count += 1


if __name__ == "__main__":
    filePath = "./tmp"
    delete_folder_and_create_new_empty_folder(filePath)
    video_file_path = download_and_save_video(filePath)
    split_video_into_frames_and_save_pngs(video_file_path, filePath + "/")




