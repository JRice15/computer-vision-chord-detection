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

vid = readvid(args.file)[:20]
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
edges = [cv.Canny(i, 10, 70) for i in blurred]
# edges = [cv.dilate(i, np.ones((3,3),np.uint8),iterations=1) for i in edges]
# edges = [cv.erode(i, np.ones((5,5),np.uint8),iterations=1) for i in edges]

# showvid(edges)

# conts = [cv.findContours(i, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)[0] for i in edges]

"""
find lines from the edges
"""
def horizontal_ish(angle):
    """
    filter angles within 45deg of horizontal
    """
    if (np.pi*1/4 < angle < np.pi*3/4) or (np.pi*1/4 < angle < np.pi*3/4):
        return True
    return False

lines = [cv.HoughLines(i, 1, np.pi/180, 100, None, 0, 0) for i in edges]
lines = [i if i is not None else [] for i in lines]

# remove mostly vertical lines and take the first 500; they should be returned in order of confidence
lines = [[j for j in framelines if horizontal_ish(j[0][1])][:500] for framelines in lines]

def findparallel(lines, theta_threshold=3, maxn=None):
    """
    args:
        theta_threshold: degrees, the threshold for how close in angle lines 
            have to be to be parallel
        maxn: number of top line bundles to find in each frame
    """
    theta_threshold = np.pi/180 * theta_threshold
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
        # sort by number of lines found in each bundle, take top maxn
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

# showvid(linesvid)

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


# showvid(boxes_vid)
# writevid(boxes_vid, "fretboard_bound")


"""
smoothing
"""
# span is number of frames on each side to average the bounding boxes of
span = 3
# num_outliers number of highest and lowest to remove before averaging
num_outliers = 0
avg_boxes = []
boxes_vid = [np.copy(i) for i in vid]

assert len(boxes) >= (2 * span + 1)
for i, box in enumerate(boxes):
    # get a selection of 2*span+1 consecutive boxes, and repeat some when at the edges of the video
    top = i + span + 1
    bottom = i-span
    if bottom < 0:
        bottom = 0
    selection = boxes[bottom:top]
    if i < span:
        selection = np.concatenate((selection, boxes[:(span-i)]), axis=0)
    elif i >= (len(boxes) - span):
        i2 = len(boxes) - i - 1
        selection = np.concatenate((selection, boxes[-(span - i2):]), axis=0)

    # remove highest and lowest n
    if num_outliers > 0:
        newboxes = []
        for box in np.array(selection).T:
            new = []
            for row in box:
                for _ in range(num_outliers):
                    row = np.delete(row, row.argmin())
                    row = np.delete(row, row.argmax())
                new.append(row)
            newboxes.append(new)
        selection = np.array(newboxes).T
        
    # average the rest
    newbox = np.mean(selection, axis=0)
    newbox = np.int0(newbox)
    avg_boxes.append(newbox)
    cv2.drawContours(boxes_vid[i],[newbox],0,(0,0,255),2)


boxes = avg_boxes
showvid(boxes_vid, name="smoothed")


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
    if len(np.squeeze(y)) == 0:
        # bad frame, no box found
        continue
    (topy, topx) = (np.min(y), np.min(x))
    (bottomy, bottomx) = (np.max(y), np.max(x))
    rot_frame = rot_frame[topy:bottomy+1, topx:bottomx+1]
    # TODO extend the bottom, since it tends to miss the smaller strings more often?
    fretboard_vid.append(rot_frame)

showvid(fretboard_vid, name="fretboard", ms=50)

"""
find vertical lines that should correspond to frets
"""
gray = [cv.cvtColor(i, cv.COLOR_BGR2GRAY) for i in fretboard_vid]
blurred = [cv.GaussianBlur(i, (5,5), 2) for i in gray]

# a bit lower threshold now, as there is less distracting noise
edges = [cv.Canny(i, 10, 50) for i in blurred]

showvid(edges)

def vertical(angle, threshhold=3):
    """
    find vertical lines within <threshhold> degrees
    """
    # convert to radians
    threshhold = np.pi/180 * threshhold
    # find vertical lines (either "up" or "down")
    if (-threshhold < angle < threshhold) or (np.pi-threshhold < angle < np.pi+threshhold):
        return True
    return False
    
lines = [cv.HoughLines(i, 1, np.pi/180, 25, None, 0, 0) for i in edges]
lines = [i if i is not None else [] for i in lines]

# find vertical lines, take first top_n (as the lines are returned in order of confidence)
top_n = 150
lines = [[j for j in framelines if vertical(j[0][1])][:top_n] for framelines in lines]

linesvid = [np.copy(i) for i in fretboard_vid]
for idx, frame_lines in enumerate(lines):
    for i in range(len(frame_lines)):
        rho = frame_lines[i][0][0]
        theta = frame_lines[i][0][1]
        a = math.cos(theta)
        b = math.sin(theta)
        x0 = a * rho
        y0 = b * rho
        pt1 = (int(x0 + 2000*(-b)), int(y0 + 2000*(a)))
        pt2 = (int(x0 - 2000*(-b)), int(y0 - 2000*(a)))
        cv.line(linesvid[idx], pt1, pt2, (255,255,255), 3, cv.LINE_AA)

showvid(linesvid, ms=100)

"""
match vertical lines to fret spacing
"""


# get position of lines horizontally (since they must be approximately vertical, the 
#  rho should be approximately the horizontal distance)
line_pos = [np.sort(np.array([j[0][0] for j in framelines])) for framelines in lines]

# ratio of fret length to length of next fret, down and up the neck
D_RATIO = 0.943874312682
U_RATIO = 1.05946309436

def match_frets(line_pos):
    """
    for each pair of lines positions
    """
    for i in range(len(line_pos) - 1):
        for j in range(i+1, i+len(line_pos[i:])):
            d = line_pos[j] - line_pos[i]



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

