import argparse
import os
import time

import cv2
import numpy as np
import pandas as pd

cv = cv2
from cv_helpers import *

"""
command line args
"""
parser = argparse.ArgumentParser()
parser.add_argument("--name",required=True,help="name to give to this file. include things like what type of guitar, and any other relevant info")
parser.add_argument("--file",default="chords.xlsx",help="chords excel file")
args = parser.parse_args()

"""
read and process file
"""
df = pd.read_excel(args.file, converters={"Name": str, "Tab": str})
# remove 'comment' lines and empty lines
df = df[~df['Name'].str.startswith("#", na=False)]
df = df.dropna(how='all')

if df.isnull().values.any():
    raise ValueError("NaN values encountered:\n\n" + df.to_string())

chords = df.to_numpy()

max_len = max([len(i) for i in chords[:,0]]) + 2

def chord_repr(c):
    return "{:{max_len}} {}".format(*c, max_len=max_len)

def random_chord():
    return chords[np.random.choice(len(chords))]

print("available chords:")
for i in chords:
    print(chord_repr(i))


"""
begin video capture
"""

print("\nA chord will show, and then a countdown from 3. On 0, play the the chord. " 
      "Then, prepare for the next. Using a larger font size in the terminal (CMD PLUS "
      "on my Mac) is probably useful")
print("\nIf you mess up, or are done, do CNTRL-C to keyboard interrupt, and it "
      "will throw away the last two chords, and save the the rest to 'raw_data/'")
print("\nStarting pre-preparation countdown. Make sure the neck of the " 
      "guitar is entirely in the frame, and then make sure you can view the terminal")

print("Press Return to continue", end=" ")
input()

start = time.time()
def elapsed():
    return time.time() - start

# I think you can change the camera here, if you have multiple
cap = cv2.VideoCapture(0)

preprep_secs = 10
countdown = 0
while elapsed() < preprep_secs:
    # Capture frame-by-frame
    ok, frame = cap.read()
    if not ok:
        print("frame failed, retrying...")
        continue

    # Display the resulting frame
    cv2.imshow('frame',frame)
    cv.waitKey(1)

    if elapsed() > countdown:
        print(preprep_secs - countdown)
        countdown += 1  


secs_per_chord = 4
frame_ind = 0
frames = []
chord_map = [] # tuples: (index, chord_fingering)
start = time.time()
countdown = 0
very_start_time = time.time()

try:
    print("Starting...")
    while(True):
        # Capture frame-by-frame
        ok, frame = cap.read()
        if not ok:
            print("! frame failed, retrying...")
            continue
        # Display the resulting frame
        cv2.imshow('frame',frame)
        if cv.waitKey(1) != -1:
            end = time.time()
            print("ending capture...")
            break
        frames.append(frame)
        frame_ind += 1

        if elapsed() > countdown:
            if countdown == 0:
                chord = random_chord()
                print("\nNext chord:", chord_repr(chord))
                print(secs_per_chord)
                countdown += 1
            elif countdown >= secs_per_chord:
                print("Play")
                chord_map.append((frame_ind, chord[1]))
                countdown = 0
                start = time.time()
            else:
                print(secs_per_chord - countdown)
                countdown += 1
except KeyboardInterrupt:
    print("\nEnding capture")

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()


if len(chord_map) <= 2:
    print("too short to do anything with. exiting...")
    exit()

# up until the last two chords
frames = frames[:chord_map[-2][0]]
chord_map = chord_map[:-2]

# save data
os.makedirs("raw_data/", exist_ok=True)
os.makedirs("data/", exist_ok=True)
filename = args.name + "_" + str(int(very_start_time))
writevid(frames, name='raw_data/'+filename)
# save npy data as well, as array of strings
print("Saving chord data")
np.save('raw_data/'+filename, np.array(chord_map))


