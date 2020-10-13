import argparse
import os
import sys

import cv2
cv = cv2
import numpy as np

from cv_helpers import *

parser = argparse.ArgumentParser()
parser.add_argument("--file",default="acoustic_light_short.mov")
args = parser.parse_args()

vid = readvid(args.file)

gray = [cv.cvtColor(i, cv.COLOR_BGR2GRAY) for i in vid]
blurred = [cv.GaussianBlur(i, (5,5), 2) for i in gray]

# thresh = [cv.adaptiveThreshold(i, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 23, 0) for i in blurred]
# thresh_vid = [
#     cv2.threshold(i, 28, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1] for i in blurred
# ]
# showvid(thresh_vid)

edges = [cv.Canny(i, 10, 100) for i in blurred]
showvid(edges)

conts = [cv.findContours(i, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)[0] for i in edges]

# print(conts[0])

v2 = []
for i, frame in enumerate(vid):
    for j in conts[i]:
        print(j)
        frame = cv.drawContours(frame, j, 0, (255,0,255))
        v2.append(frame)

showvid(v2)