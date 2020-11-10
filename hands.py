import argparse
import math
import os
import sys
import time

import cv2
cv = cv2

import matplotlib.pyplot as plt
import mediapipe as mp
import numpy as np

from cv_helpers import *

"""
mediapipe hand detection
"""

parser = argparse.ArgumentParser()
parser.add_argument("--file", required=True)
args = parser.parse_args()

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands


vid = readvid(args.file, maxframes=100)

# For webcam input:
hands = mp_hands.Hands(
    min_detection_confidence=0.1, 
    min_tracking_confidence=0.7)


shapey, shapex, shapez = vid[0].shape

handvid = []
for i, image in enumerate(vid):
    # Flip the image horizontally for a later selfie-view display, and convert
    # the BGR image to RGB.
    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    results = hands.process(image)

    # Draw the hand annotations on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            count = 0
            x = int(np.median([l.x for l in hand_landmarks.landmark]) * shapex)
            y = int(np.median([l.y for l in hand_landmarks.landmark]) * shapey)
            for l in hand_landmarks.landmark:
                count += 1
                p = (int(l.x * shapex), int(l.y * shapey))
                image = cv.circle(image, p, 4, (255,0,0), thickness=cv.FILLED, lineType=cv.FILLED)
            image = cv.circle(image, (x,y), 6, (0,255,0), thickness=cv.FILLED, lineType=cv.FILLED)
            mp_drawing.draw_landmarks(
                image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
    handvid.append(cv.flip(image, 1))


hands.close()

showvid(handvid, ms=60)

mp.framework.formats.landmark_pb2.NormalizedLandmark
