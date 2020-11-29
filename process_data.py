import os
import sys
import argparse
import traceback

from src import detect_fretboard
from src.cv_helpers import *


parser = argparse.ArgumentParser()
parser.add_argument("--file",default=None,help="specify one specific file in 'raw_data' to process")
parser.add_argument("--overwrite",action="store_true",help="overwrite all fretboard videos in the 'data/' dir")
args = parser.parse_args()


def remove_str_prefix(s, prefix):
    if s.startswith(prefix):
        return s[len(prefix):]
    raise ValueError("'"+s+"' doesn't have the prefix '"+prefix+"'")

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
    # compare raw_data to data, get those in raw_data that aren't in data already
    to_process = [i for i in raw if split_path(i)[1] not in data]

if args.file is not None:
    file = remove_str_prefix(args.file, "raw_data/")
    if file in to_process:
        to_process = [file]
    else:
        raise ValueError("File '" + file + "' is not in to_process queue. Does it need the '--overwrite' argument also?")

"""
process them
"""

if not to_process:
    print("no new raw_data to process...")
    exit()

successes = 0
failures = 0
for name in to_process:
    try:
        print("\n*** Processing", name)
        detect_fretboard.main(
            file="raw_data/"+name, 
            outfile="data/"+name, 
            full=True,
            nofrets=True,
            show=False
        )
        successes += 1
    except Exception as e:
        print("\n*** Exception:")
        traceback.print_exc()
        print("\n")
        failures += 1
    
print()
print(successes, "successes,", failures, "failures")


