import argparse
import os
import time

import cv2
import numpy as np
import pandas as pd
import multiprocessing

cv = cv2
from src.cv_helpers import *

DONE_FLAG = "DONE"

os.makedirs("data/image_model_train", exist_ok=True)
os.makedirs("data/inference_model_train", exist_ok=True)
os.makedirs("data/inference_model_test", exist_ok=True)

def chord_repr(c, max_len):
    return "{:{max_len}} {}".format(*c, max_len=max_len)

def random_chord(chords):
    return chords[np.random.choice(len(chords))]

def read_chordfile(args):
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
    
    return chords


def prep_capture(cap):
    """
    preparation time
    """
    print("\nA chord will show, and then a countdown from 3. On 0, play the the chord. " 
        "Then, prepare for the next. Using a larger font size in the terminal (CMD PLUS "
        "on my Mac) is probably useful")
    print("\nIf you mess up, or are done, do CNTRL-C to keyboard interrupt, and it "
        "will throw away the last two chords, and save the the rest to 'raw_data/'")
    print("\nStarting pre-preparation countdown. Make sure the neck of the " 
        "guitar is entirely in the frame, and then make sure you can view the terminal")

    print("Do a keyboard interrupt (CTRL-C) to continue")

    try:
        while True:
            # Capture frame-by-frame
            ok, frame = cap.read()
            if not ok:
                print("frame failed, retrying...")
                continue

            # Display the resulting frame
            cv2.imshow('frame',frame)
            cv.waitKey(1)

    except KeyboardInterrupt:
        print("Starting...")
        return

# thanks https://stackoverflow.com/questions/842557/how-to-prevent-a-block-of-code-from-being-interrupted-by-keyboardinterrupt-in-py
import signal
import logging

class DelayedKeyboardInterrupt(object):
    def __enter__(self):
        self.signal_received = False
        self.old_handler = signal.signal(signal.SIGINT, self.handler)

    def handler(self, sig, frame):
        self.signal_received = (sig, frame)
        # logging.debug('SIGINT received. Delaying KeyboardInterrupt.')

    def __exit__(self, type, value, traceback):
        signal.signal(signal.SIGINT, self.old_handler)
        if self.signal_received:
            self.old_handler(*self.signal_received)

def process_task(filename, q):
    """
    process to run in seperate multiprocessing thread
    """
    try:
        with DelayedKeyboardInterrupt():
            writer = skvideo.io.FFmpegWriter(
                "raw_data/" + filename + ".mp4", 
                outputdict={"-pix_fmt": "yuv420p"},
                # verbosity=1
            )
            while True:
                frames = q.get(block=True, timeout=20)
                if frames == DONE_FLAG:
                    break
                for frame in frames:
                    writer.writeFrame(frame[...,::-1])
            writer.close()
    except KeyboardInterrupt:
        print("Background process done")


def run_capture(args, cap, chords):
    """
    run main capture, using multiprocessing to write out video as we go
    """
    max_chord_len = max([len(i) for i in chords[:,0]]) + 2

    os.makedirs("raw_data/", exist_ok=True)
    capture_time = time.time()
    filename = args.name + "_" + str(int(capture_time))

    # multiprocessing to write out file in seperate thread
    ctx = multiprocessing.get_context("spawn")
    queue = ctx.Queue()
    process = ctx.Process(target=process_task, args=(filename, queue))
    process.start()

    secs_per_chord = 4
    frame_ind = 0
    frame_sets = [[]]
    chord_map = [] # tuples: (index, chord_fingering)
    chord_ind = 0
    countdown = 0

    start = time.time()
    def elapsed():
        return time.time() - start

    try:
        print("Starting...")
        while cv.waitKey(1) != "q":
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
            # if no chords yet, dont save frame
            if chord_map:
                # add frame to last frame set
                frame_sets[-1].append(frame)
                frame_ind += 1

            if elapsed() > countdown:
                if countdown == 0:
                    chord = random_chord(chords)
                    print("\nNext chord:", chord_repr(chord, max_chord_len))
                    print(secs_per_chord)
                    countdown += 1
                elif countdown >= secs_per_chord:
                    print("Play")
                    chord_map.append((frame_ind, chord[1]))
                    chord_ind += 1
                    countdown = 0
                    start = time.time()
                    if chord_ind > 3:
                        # add to write queue
                        queue.put_nowait(frame_sets[0])
                        frame_sets.pop(0)
                        frame_sets.append([])
                else:
                    print(secs_per_chord - countdown)
                    countdown += 1
    except KeyboardInterrupt:
        print("\nEnding capture")

    # everything is done, release things
    queue.put(DONE_FLAG)
    queue.close()
    process.join()
    cap.release()
    cv2.destroyAllWindows()

    if len(chord_map) <= 3:
        print("too short to do anything with. exiting...")
    else:
        # save npy data, as array of strings
        print("Saving chord data")
        os.makedirs("data/", exist_ok=True)
        np.save('data/'+filename, np.array(chord_map))



def main():
    """
    command line args
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--name",required=True,help="name to give to this file. include things like what type of guitar, and any other relevant info")
    parser.add_argument("--file",default="chords.xlsx",help="chords excel file")
    parser.usage = parser.format_help()
    args = parser.parse_args()

    chords = read_chordfile(args)

    max_chord_len = max([len(i) for i in chords[:,0]]) + 2

    print("available chords:")
    for i in chords:
        print(chord_repr(i, max_chord_len))

    # I think you can change the camera here, if you have multiple
    cap = cv2.VideoCapture(0)

    prep_capture(cap)

    run_capture(args, cap, chords)

print(multiprocessing.current_process())

if __name__ == "__main__":
    main()