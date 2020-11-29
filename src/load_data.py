import numpy as np
import os
from src.cv_helpers import *


def ok_file(filename):
    if filename.startswith("."):
        return False
    if filename.endswith(".py"):
        return False
    if filename.startswith("__"):
        return False
    return True

def str_to_chord(s):
    return np.array(list(s), dtype=int)

def get_all_data_names():
    xnames = []
    ynames = []

    # mapping name to extension
    found_exts = {}

    # find which paths to load, by matching npy files with video files
    for filename in os.listdir("data"):
        if ok_file(filename):
            _, name, ext = split_path(filename)
            if ext == ".npy":
                if name in found_exts:
                    if found_exts[name] != ".npy":
                        xnames.append("data/" + name + found_exts[name])
                        ynames.append("data/" + filename)
                    else:
                        raise ValueError("Multiple .npy with name '" + filename + "'")
                else:
                    found_exts[name] = ext
            elif ext in (".mov", ".mp4"):
                if name in found_exts:
                    if found_exts[name] == ".npy":
                        xnames.append("data/" + filename)
                        ynames.append("data/" + name + found_exts[name])
                    else:
                        raise ValueError("Multiple vid with name '" + filename + "'")
                else:
                    found_exts[name] = ext
            else:
                raise ValueError("Unknown filetype in 'data' dir: '" + filename + "'")
    
    return xnames, ynames


def load_raw_files(xnames, ynames, do_test=False, display=False, y_only=False):
    # y is list of arrays, each array is a chord map [ [frame index, chord str], ... ]
    yraw = []
    # x is a list of arrays, each array is a video, with shape (numframes, x, y, 3)
    xraw = []

    for i in range(len(xnames)):
        yraw.append(np.load(ynames[i]))
        vid = readvid(xnames[i], maxframes=(30 if do_test else None))
        print(len(vid), "frames")
        vid = preprocess_vid(vid, display=display)
        xraw.append(vid)

    if len(yraw) < 1 or (len(xraw) < 1 and not y_only):
        raise ValueError("No matching data in the 'data' directory!")
    
    return xraw, yraw

def expand_chords(xraw, yraw, no_transitions=False):
    xs = []
    ys = []

    # for each video loaded, match video frames to the correct chords
    for i in range(len(yraw)):
        start = int(yraw[i][0][0]) # first chord, its index
        thisx = []
        thisy = []
        # frame index, relative to video
        vid_ind = start
        # index of next upcoming chord in y[i]
        y_ind_next = 0
        y_ind_curr = -1
        while vid_ind < len(xraw[i]):
            # if the frameindex is at the next chord in y, move up the pointers
            if y_ind_next < len(yraw[i]) and int(yraw[i][y_ind_next][0]) == vid_ind:
                chord = str_to_chord(yraw[i][y_ind_next][1])
                y_ind_next += 1
                y_ind_curr += 1
                vid_ind_curr = int(yraw[i][y_ind_curr][0])
            # get where we are in this chord
            if y_ind_next < len(yraw[i]):
                chord_len = int(yraw[i][y_ind_next][0]) - int(yraw[i][y_ind_curr][0])
            else:
                chord_len = len(xraw[i]) - int(yraw[i][y_ind_curr][0])
            # only keep frames solidly in the middle of a chord (no transitions)
            pos_in_chord = vid_ind - vid_ind_curr
            if (not no_transitions) or not ((pos_in_chord / chord_len < 0.2) or (pos_in_chord / chord_len > 0.8)):
                thisy.append(chord)
                thisx.append(xraw[i][vid_ind])
            vid_ind += 1
        ys.append(thisy)
        xs.append(thisx)

    # make sure everythig is the same size
    assert len(xs) == len(ys)
    for i in range(len(xs)):
        if len(xs[i]) != len(ys[i]):
            print("xs", i, "len", len(xs[i]))
            print("ys", i, "len", len(ys[i]))
            raise ValueError()
    
    return xs, ys


def preprocess_vid(vid, display=False):
    """
    height, width: size
    """
    IM_HEIGHT = 108
    IM_WIDTH = 540
    vid = [cv.resize(im, dsize=(IM_WIDTH,IM_HEIGHT), interpolation=cv.INTER_AREA) for im in vid]
    if display:
        showim(vid[0], ms=400)
    return vid


def train_val_test(x, split=0.15):
    length = len(x)
    testsplit = -int(split * length)
    valsplit = -int(split * (length + testsplit)) + testsplit
    xtest = x[testsplit:]
    xval = x[valsplit:testsplit]
    xtrain = x[:valsplit]
    return xtrain, xval, xtest
    

def load_one(xname, yname, display=False, y_only=False, no_transitions=False):
    """
    args:
        xname, yname: filepath to x and y data
        other args: see load_all_data docstring
    """
    xraw, yraw = load_raw_files([xname], [yname], display=display, y_only=y_only)
    xs, ys = expand_chords(xraw, yraw, no_transitions=no_transitions)
    y = np.array(ys[0])
    if y_only:
        return y
    x = np.array(xs[0])
    return x, y


def load_all_data(display=False, do_test=False, y_only=False, no_transitions=True):
    """
    args:
        display: whether to show some example ims while loading
        do_test: whether to only load a test portion of the data
        y_only: whether only the ydata is needed
        no_transistions: whether to cut off the transitions between chords
    returns:
        if y_only:
            ytrain, yval, ytest
        else:
            xtrain, xval, xtest, ytrain, yval, ytest
    """

    xnames, ynames = get_all_data_names()

    xraw, yraw = load_raw_files(xnames, ynames, do_test=do_test, display=display, y_only=y_only)

    xs, ys = expand_chords(xraw, yraw, no_transitions=no_transitions)

    # free up memory
    del xraw, yraw

    print(len(xs), "data videos found")

    """
    splitting
    """

    print("making train/val/test splits")
    xtrain, xval, xtest = [], [], []
    ytrain, yval, ytest = [], [], []

    for i in range(len(xs)):
        split = train_val_test(ys[i], split=0.15)
        ytrain += split[0]
        yval += split[1]
        ytest += split[2]
        if not y_only:
            split = train_val_test(xs[i], split=0.15)
            xtrain += split[0]
            xval += split[1]
            xtest += split[2]

    # free up memory
    del split, ys
    if not y_only: del xs

    # convert to numpy
    ytrain = np.array(ytrain)
    yval = np.array(yval)
    ytest = np.array(ytest)
    if y_only:
        return ytrain, yval, ytest
    xtrain = np.array(xtrain)
    xval = np.array(xval)
    xtest = np.array(xtest)

    if display:
        showvid(xtrain[:50], name="xtrain", ms=25)

    print("img_shape", xtrain[0].shape)

    return xtrain, xval, xtest, ytrain, yval, ytest