import cv2
from typing import List
import numpy as np
from PIL import Image
import os


def get_frames(videopath: str, frames_dir: str, size_format: str, max_image_length: int = -1,
               start_frame: int = -1,
               end_frame: int = -1, overwrite=True) -> List[str]:
    capture = cv2.VideoCapture(videopath)

    if start_frame < 0: start_frame = 0  # if start isn't specified lets assume 0
    if end_frame < 0: end_frame = int(
        capture.get(cv2.CAP_PROP_FRAME_COUNT))  # if end isn't specified assume the end of the video

    capture.set(1, start_frame)  # set the starting frame of the capture
    current_frame = start_frame  # keep track of which frame we are up to, starting from start
    while_safety = 0  # a safety counter to ensure we don't enter an infinite while loop (hopefully we won't need it)
    processed_count = 0  # a count of how many frames we have saved
    frame_paths = []  # resized to max_image_length (if > 0)

    while current_frame <= end_frame:  # loop through the frames until the end
        _, image = capture.read()  # read an image from the capture
        if while_safety > 500: break  # break the while if our safety maxs out at 500
        if image is None:  # if we get a bad return flag or the image we read is None, lets not save
            while_safety += 1  # add 1 to our while safety, since we skip before incrementing our frame variable
            continue  # skip

        while_safety = 0
        save_path = os.path.join(frames_dir, "{:010d}.jpg".format(processed_count))  # create the save path
        if not os.path.exists(save_path) or overwrite:  # if it doesn't exist or we want to overwrite anyways
            if max_image_length > 0:
                image = resize_aspect_ratio(image, size_format, max_image_length)

            cv2.imwrite(save_path, image)  # save the extracted image
            processed_count += 1  # increment our counter by one
        frame_paths.append(save_path)
        current_frame += 1
    capture.release()  # after the while has finished close the capture
    return frame_paths


def image_resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image

    resized = cv2.resize(image, dim, interpolation=inter)

    # return the resized image
    return resized


def resize_aspect_ratio(image: np.array, size_format: str, max_image_length: int) -> np.array:
    if size_format in ["horizontal", "square"]:
        image = image_resize(image, width=max_image_length)
    else:
        image = image_resize(image, height=max_image_length)

    return image


def get_video_format(video_path: str):
    capture = cv2.VideoCapture(video_path)
    fps = int(capture.get(cv2.CAP_PROP_FPS))
    _, frame = capture.read()
    capture.release()

    frame = Image.fromarray(frame)
    size_orig = frame.size
    if size_orig[0] > size_orig[1]:
        size_format = "horizontal"
    elif size_orig[0] < size_orig[1]:
        size_format = "vertical"
    else:
        size_format = "square"

    return size_format, fps