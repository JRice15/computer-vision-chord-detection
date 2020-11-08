import cv2
cv = cv2
import skvideo.io
import numpy as np
import os

"""
basic helper functions
"""

def split_path(path):
    root, file = os.path.split(path)
    name, ext = os.path.splitext(file)
    return root, name, ext

def strip_extension(name):
    reverse_ind = name[::-1].find(".")
    if reverse_ind != -1:
        name = name[:-reverse_ind-1]
    return name

def readvid(file):
    vid = cv.VideoCapture(file)
    vidframes = []
    while vid.isOpened():
        ok, frame = vid.read()
        if not ok:
            break
        vidframes.append(frame)
    return vidframes

def writevid(vid, name, flipchannels=True):
    """
    name: filename to save under
    flipchannels: bool, true if video is in BGR
    """
    name = strip_extension(name)
    print("writing vid", name+".mp4 ...")
    vid = np.array(vid)
    vid = vid[...,::-1]
    skvideo.io.vwrite(name+".mp4", vid, outputdict={"-pix_fmt": "yuv420p"})

def showim(img, name="", ms=1000):
    """
    show image with a good wait time
    """
    cv2.imshow(name, img)
    cv.moveWindow(name, 0, 0)
    cv2.waitKey(ms)
    cv.destroyWindow(name)
    cv.waitKey(1)

def showvid(vid, name="", ms=25):
    """
    show vid, press a key to cancel
    """
    for frame in vid:
        cv.imshow(name, frame)
        cv.moveWindow(name, 0, 0)
        if cv.waitKey(ms) != -1:
            break
    cv.destroyWindow(name)
    cv.waitKey(1)


