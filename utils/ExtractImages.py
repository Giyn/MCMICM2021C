"""
-------------------------------------
# -*- coding: utf-8 -*-
# @Time    : 2021/2/5 15:37:37
# @File    : ExtractImages.py
# @Software: PyCharm
-------------------------------------
"""

import os
import cv2


def save_image(num: int, image, folder: str):
    """Save the images.

    Args:
        num   : serial number
        image : image resource
        folder: folder name

    Returns:
        None
    """
    image_path = '../data/images_data/{}/{}.png'.format(folder, str(num))
    if image is not None:
        cv2.imwrite(image_path, image)


def extract_image(path: str, img_folder: str):
    """Extract images.

    Args:
        path      : video path
        img_folder: image folder name

    Returns:
        None
    """
    vc = cv2.VideoCapture('../data/raw_media_files/{}'.format(path))  # import video files

    # determine whether to open normally
    if vc.isOpened():
        ret, frame = vc.read()
    else:
        ret = False

    count = 0  # count the number of pictures
    frame_interval = 20  # video frame count interval frequency
    frame_interval_count = 0

    # loop read video frame
    while ret:
        ret, frame = vc.read()
        # store operation every time f frame
        if frame_interval_count % frame_interval == 0:
            save_image(count, frame, img_folder)
            count += 1
        frame_interval_count += 1
        cv2.waitKey(1)

    vc.release()


if __name__ == '__main__':
    video_file_list = [i for i in os.listdir('../data/raw_media_files') if
                       ('.MOV' in i) or ('.mov' in i) or ('.MP4' in i) or ('.mp4' in i)]
    for j in video_file_list:
        print(j)
        folder_name = j.split('_')[0]
        os.mkdir('../data/images_data/{}'.format(folder_name))
        extract_image(j, folder_name)
