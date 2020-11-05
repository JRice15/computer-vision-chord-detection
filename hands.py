import argparse
import math
import os
import sys
import time

import cv2
cv = cv2
import matplotlib.pyplot as plt

import numpy as np

from cv_helpers import *


# mediapipe hand detection

import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands


vid = readvid("fullvids/acoustic_light.mov")

# For webcam input:
hands = mp_hands.Hands(
    min_detection_confidence=0.3, 
    min_tracking_confidence=0.3)

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
            mp_drawing.draw_landmarks(
                image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
    handvid.append(cv.flip(image, 1))

hands.close()

showvid(handvid, ms=35)

