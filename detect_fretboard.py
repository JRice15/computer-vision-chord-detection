import argparse
import math
import os
import sys

import cv2
cv = cv2
import matplotlib.pyplot as plt

import numpy as np

from cv_helpers import *

parser = argparse.ArgumentParser()
parser.add_argument("--file",default="acoustic_light_short.mov")
args = parser.parse_args()

vid = readvid(args.file)[:40]
# add neck image as last in pipeline
# vid.append(cv.imread("neck_dark.jpg"))

black = np.zeros((len(vid),)+vid[0].shape)

"""
grayscale and blur
"""
gray = [cv.cvtColor(i, cv.COLOR_BGR2GRAY) for i in vid]
blurred = [cv.GaussianBlur(i, (5,5), 2) for i in gray]

# thresh = [cv.adaptiveThreshold(i, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 23, 0) for i in blurred]
# thresh_vid = [
#     cv2.threshold(i, 28, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1] for i in blurred
# ]
# showvid(thresh_vid)

"""
edge detection. the second number is the main threshold (lower == more edges, more noise)
"""
edges = [cv.Canny(i, 10, 50) for i in blurred]
# edges = [cv.dilate(i, np.ones((3,3),np.uint8),iterations=1) for i in edges]
# edges = [cv.erode(i, np.ones((5,5),np.uint8),iterations=1) for i in edges]
showvid(edges)

# conts = [cv.findContours(i, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)[0] for i in edges]

"""
find lines from the edges
"""
lines = [cv.HoughLines(i, 1, np.pi/180, 100, None, 0, 0) for i in edges]

def findparallel(lines, theta_threshold=(np.pi/180*3), maxn=None):
    """
    theta_threshold: radians, the threshold for how close lines have to be to be parallel
    """
    vid_lines = []
    for frame_lines in lines:
        all_lines = []
        for i, l1 in enumerate(frame_lines):
            this_lines = [l1]
            for j, l2 in enumerate(frame_lines):
                if (i == j):
                    continue
                if (abs(l1[0][1] - l2[0][1]) <= theta_threshold):          
                    this_lines.append(l2)
            all_lines.append(this_lines)
        """
        sort by number of lines found in each bundle, take top maxn
        """
        sorted(all_lines, key=lambda x: len(x), reverse=True)
        if maxn is not None:
            all_lines = all_lines[:maxn]
        joined = []
        for x in all_lines:
            joined += x
        vid_lines.append(joined)
    return vid_lines

lines = findparallel(lines, maxn=1)

"""
write the lines onto a black video
"""
# linesvid = [np.copy(i) for i in vid]
linesvid = black

if lines is not None:
    for idx, frame_lines in enumerate(lines):
        for i in range(0, len(frame_lines)):
            rho = frame_lines[i][0][0]
            theta = frame_lines[i][0][1]
            a = math.cos(theta)
            b = math.sin(theta)
            x0 = a * rho
            y0 = b * rho
            pt1 = (int(x0 + 2000*(-b)), int(y0 + 2000*(a)))
            pt2 = (int(x0 - 2000*(-b)), int(y0 - 2000*(a)))
            cv.line(linesvid[idx], pt1, pt2, (255,255,255), 3, cv.LINE_AA)

linesvid = [np.uint8(cv.cvtColor(np.float32(i), cv.COLOR_BGR2GRAY)) for i in linesvid]

showvid(linesvid)

"""
find the contours of the linesvid. It should find a big bounding box around the big
cluster of the fretboards, so we take the contour with the biggest area
"""
boxes_vid = [np.copy(i) for i in vid]

boxes = []
for i, frame in enumerate(linesvid):
    contours, heirarchy = cv2.findContours(frame, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    im = boxes_vid[i]
    # get largest contour
    cnt = max(contours, key=cv.contourArea)
    # find its (potentially nonvertical) bounding rect
    rect = cv2.minAreaRect(cnt)
    box = cv.boxPoints(rect)
    box = np.int0(box)
    cv2.drawContours(im,[box],0,(0,0,255),2)
    boxes.append(box)

showvid(boxes_vid)
# writevid(boxes_vid[:-1], "fretboard_bound")

"""
rotate the frames so that the fretboard is horizontal
"""
def rotate_image(image, angle):
  image_center = tuple(np.array(image.shape[1::-1]) / 2)
  rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
  result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
  return result


fretboard_vid = []
for i, frame in enumerate(vid):
    radians = np.mean([l[0][1] for l in lines[i]])
    degrees = np.degrees(radians) - 90
    mask = np.zeros(frame.shape)
    cv2.drawContours(mask, [boxes[i]], 0, (255,255,255), -1)

    rot_frame = rotate_image(frame, degrees)
    rot_mask = rotate_image(mask, degrees)

    y,x,z = np.nonzero(rot_mask)
    (topy, topx) = (np.min(y), np.min(x))
    (bottomy, bottomx) = (np.max(y), np.max(x))
    rot_frame = rot_frame[topy:bottomy+1, topx:bottomx+1]
    fretboard_vid.append(rot_frame)


showvid(fretboard_vid, name="fretboard", ms=50)



"""
tried some keypoint matching, but didn't make much progress
"""
### keypoint matching

# neck = vid[-1]
# vid = vid[:-1]

# detector = cv.ORB_create()
# # find the keypoints and descriptors with ORB

# kp1, des1 = detector.detectAndCompute(neck,None)
# matches_vid = []
# for frame in vid:
#     kp2, des2 = detector.detectAndCompute(frame,None)

#     matcher = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
#     # Match descriptors.
#     matches = matcher.match(des1,des2)
#     # Sort them in the order of their distance.
#     matches = sorted(matches, key = lambda x:x.distance)
#     # Draw first 10 matches.
#     img3 = cv.drawMatches(neck,kp1,frame,kp2,matches,None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
#     matches_vid.append(img3)

# showvid(matches_vid, ms=25)

