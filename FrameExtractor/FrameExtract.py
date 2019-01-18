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
    bunnyVideo = YouTube("https://www.youtube.com/watch?v=goSVynBVnLs")
    bunnyVideo = bunnyVideo.streams.first()
    filePath = filePath + "/"
    bunnyVideo.download(filePath)
    return filePath + bunnyVideo.default_filename


def split_video_into_frames_and_save_pngs(filePath):
    vidcap = cv2.VideoCapture(filePath)
    success, image = vidcap.read()
    count = 0
    while success:
        cv2.imwrite(filePath + "frame%d.jpg" % count, image)
        success, image = vidcap.read()
        print('Read a new frame: ', success)
        count += 1

if __name__ == "__main__":
    filePath = "./tmp"
    delete_folder_and_create_new_empty_folder(filePath)
    filePath = download_and_save_video(filePath)
    split_video_into_frames_and_save_pngs(filePath)




