import cv2
cv = cv2
import skvideo.io
import numpy as np

"""
basic helper functions
"""

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
    name: filename to save under (do not include extension)
    flipchannels: bool, true if video is in BGR
    """
    print("writing vid", name+".mp4 ...")
    vid = np.array(vid)
    vid = vid[...,::-1]
    skvideo.io.vwrite(name+".mp4", vid, outputdict={"-pix_fmt": "yuv420p"})

def showim(img, name="", ms=1000):
    """
    show image with a good wait time
    """
    cv2.imshow(name, img)
    cv2.waitKey(ms)
    cv.destroyWindow(name)
    cv.waitKey(1)

def showvid(vid, name="", ms=25):
    """
    show vid, press a key to cancel
    """
    for frame in vid:
        cv.imshow(name, frame)
        if cv.waitKey(ms) != -1:
            break
    cv.destroyWindow(name)
    cv.waitKey(1)


