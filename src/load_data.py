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

def load_data(display=False, do_test=False, y_only=False):
    """
    args:
        display: whether to show some example ims while loading
        do_test: whether to only load a test portion of the data
        y_only: whether only the ydata is needed
    returns:
        if y_only:
            ytrain, yval, ytest
        else:
            xtrain, xval, xtest, ytrain, yval, ytest
    """
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
                        xnames.append(name + found_exts[name])
                        ynames.append(filename)
                    else:
                        raise ValueError("Multiple .npy with name '" + filename + "'")
                else:
                    found_exts[name] = ext
            elif ext in (".mov", ".mp4"):
                if name in found_exts:
                    if found_exts[name] == ".npy":
                        xnames.append(filename)
                        ynames.append(name + found_exts[name])
                    else:
                        raise ValueError("Multiple vid with name '" + filename + "'")
                else:
                    found_exts[name] = ext
            else:
                raise ValueError("Unknown filetype in 'data' dir: '" + filename + "'")
        


    # y is list of arrays, each array is a chord map [ [frame index, chord str], ... ]
    yraw = []
    # x is a list of arrays, each array is a video, with shape (numframes, x, y, 3)
    xraw = []

    for i in range(len(xnames)):
        yraw.append(np.load("data/"+ynames[i]))
        vid = readvid("data/"+xnames[i], maxframes=(100 if do_test else None))
        print(len(vid), "frames")
        xraw.append(vid)

    if len(yraw) < 1 or (len(xraw) < 1 and not y_only):
        print("No data in the 'data' directory")
        exit()

    xs = []
    ys = []

    def str_to_chord(s):
        return np.array(list(s), dtype=int)

    # for each video loaded
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
            if not ((pos_in_chord / chord_len < 0.2) or (pos_in_chord / chord_len > 0.8)):
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

    # free up memory
    del xraw, yraw

    print(len(xs), "data videos found")

    """
    preprocessing
    """

    xtrain, xval, xtest = [], [], []
    ytrain, yval, ytest = [], [], []

    if not y_only:
        print("Processing videos: ", end="")
        IM_HEIGHT = 108
        IM_WIDTH = 540
        print("target height/width ratio:", IM_HEIGHT/IM_WIDTH)

    for i in range(len(xs)):
        y = ys[i]
        if not y_only:
            print(i, end=" ", flush=True)
            x = [cv.resize(im, dsize=(IM_WIDTH,IM_HEIGHT), interpolation=cv.INTER_AREA) for im in xs[i]]
            if display:
                showim(x[0], ms=400)

        # use last 15% of each video as test set
        testsplit = -int(0.15 * len(y))
        if not y_only:
            xtest += x[testsplit:]
            x = x[:testsplit]
        ytest += y[testsplit:]
        y = y[:testsplit]

        # use last 15% of remaining video (excluding test set) as validation
        valsplit = -int(0.15 * len(y))
        if not y_only:
            xtrain += x[:valsplit]
            xval += x[valsplit:]
        ytrain += y[:valsplit]
        yval += y[valsplit:]

    print()
    # free up memory
    del y, ys
    if not y_only: del x, xs

    # convert to numpy
    xtrain = np.array(xtrain)
    xval = np.array(xval)
    xtest = np.array(xtest)
    ytrain = np.array(ytrain)
    yval = np.array(yval)
    ytest = np.array(ytest)

    if y_only:
        return ytrain, yval, ytest

    if display:
        showvid(xtrain[:50], name="x", ms=25)

    print("img_shape", xtrain[0].shape)

    return xtrain, xval, xtest, ytrain, yval, ytest
