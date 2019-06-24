"""
This script intended to have the helper functions for processing video
To the project, in help with cv2
"""
import cv2
import os, operator


def key(full_name):
    name, _ = full_name.split(".")
    return int(name)


"""
function that extract the video frames

@path : video full or relevant path
@target_folder_path : Target folder to save every frame

"""

def extract_frames(path, target_folder_path):
    vidcap = cv2.VideoCapture(path)
    success, image = vidcap.read()
    count = 1
    success = True
    while success:
        cv2.imwrite(target_folder_path + "/%d.jpg" % count, image)  # save frame as JPEG file
        success, image = vidcap.read()
        count = count + 1

    print('success reading ' + str(count) + ' frames')


"""
function that combine the video into frames

@target_path : path for the video that will be generated
@frames_folder : Target folder of frames to read

"""


def combine_frames(target_path, frames_folder):
    image_folder = frames_folder
    video_name = target_path

    images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
    #images = sorted(images, key=key)
    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape

    video = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*'XVID'), 30, (width,height))

    for image in images:
        video.write(cv2.imread(os.path.join(image_folder, image)))

    cv2.destroyAllWindows()
    video.release()