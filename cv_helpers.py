import cv2
cv = cv2

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

def showim(img, name="", ms=1000):
    """
    show image with a good wait time
    """
    cv2.imshow(name, img)
    cv2.waitKey(ms)

def showvid(vid, name="", ms=25):
    """
    show vid, press a key to cancel
    """
    for frame in vid:
        cv.imshow(name, frame)
        if cv.waitKey(ms) != -1:
            break


