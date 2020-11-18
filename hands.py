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


def detect_hands(vid, show_result=True):
    """
    mediapipe hand detection
    returns:
        handpos list of (x,y) pts of approx hand locations
    """
    print("Detecting hands")
    mp_drawing = mp.solutions.drawing_utils
    mp_hands = mp.solutions.hands

    hands = mp_hands.Hands(
        min_detection_confidence=0.1, 
        min_tracking_confidence=0.7)

    shapey, shapex, shapez = vid[0].shape

    handvid = []
    handpos = []
    for i, image in enumerate(vid):
        # Flip the image horizontally for a later selfie-view display, and convert
        # the BGR image to RGB.
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        image.flags.writeable = False
        results = hands.process(image)

        if results.multi_hand_landmarks:
            xs = np.array(
                [np.median([l.x for l in hand_landmarks.landmark]) for hand_landmarks in results.multi_hand_landmarks]
            )
            ys = np.array(
                [np.median([l.y for l in hand_landmarks.landmark]) for hand_landmarks in results.multi_hand_landmarks]
            )
            xs = np.int16(xs * shapex)
            ys = np.int16(ys * shapey)
            pts = [(x, ys[i]) for i,x in enumerate(xs)]
            handpos.append(pts)

            if show_result:
                # Draw the hand annotations on the image.
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                i = 0
                colors = [(255,0,0),(0,255,0),(0,0,255)]
                for hand_landmarks in results.multi_hand_landmarks:
                    for l in hand_landmarks.landmark:
                        p = (int(l.x * shapex), int(l.y * shapey))
                        image = cv.circle(image, p, 4, colors[i], thickness=cv.FILLED, lineType=cv.FILLED)
                    i = (i + 1) % len(colors)
                for p in pts:
                    image = cv.circle(image, p, 8, (0,255,0), thickness=cv.FILLED, lineType=cv.FILLED)
                # mp_drawing.draw_landmarks(
                #     image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                handvid.append(image)
        else:
            handpos.append(None)

    hands.close()

    if show_result:
        showvid(handvid, ms=80)

    return handpos



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", required=True)
    args = parser.parse_args()
    vid = readvid(args.file, maxframes=600)
    detect_hands(vid, show_result=True)