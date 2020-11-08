import os
import sys
import argparse

import detect_fretboard
from cv_helpers import *


parser = argparse.ArgumentParser()
parser.add_argument("--overwrite",action="store_true",help="overwrite all fretboard videos in the 'data/' dir")
args = parser.parse_args()


"""
get filenames to process
"""

raw = [i for i in os.listdir("raw_data/") if not i.startswith('.')]
data = [i for i in os.listdir("data/") if not i.startswith('.')]

if args.overwrite:
    to_process = raw
else:
    # ignore npy extension
    data = [i for i in data if split_path(i)[2] != ".npy"]
    # get only filename
    data = [split_path(i)[1] for i in data]

    to_process = [i for i in raw if split_path(i)[1] not in data]


"""
process them
"""

if not to_process:
    print("no new raw_data to process...")

for name in to_process:

    print("\n*** Processing", name)
    detect_fretboard.main(
        file="raw_data/"+name, 
        outfile="data/"+name, 
        full=False,
        nofrets=True,
        show=False
    )

