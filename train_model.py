import argparse
import json
import os
import time
import pprint

import keras
import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.callbacks import (History, LearningRateScheduler, ModelCheckpoint,
                             ReduceLROnPlateau, EarlyStopping)
from keras.models import Model
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split

from cv_helpers import *
from models import make_model
from save_stats import save_history

parser = argparse.ArgumentParser()
parser.add_argument("--name",required=True)
parser.add_argument("--load",action="store_true")
parser.add_argument("--nodisplay",action="store_true")
args = parser.parse_args()

class TrainConfig:

    def __init__(self, epochs, model, batchsize, lr, lr_sched_freq, 
            lr_sched_factor, loss):
        self.epochs = epochs
        self.model = model
        self.batchsize = batchsize
        self.lr = lr
        self.lr_sched_freq = lr_sched_freq
        self.lr_sched_factor = lr_sched_factor
        self.loss = loss
        pprint.pprint(vars(self))
    
    def __str__(self):
        return str(vars(self))
    
    def write_to_file(self,filename):
        with open(filename, "a") as f:
            f.write("\n" + str(self) + "\n\n")

with open("model_config.json", "r") as f:
    config_dict = json.load(f)

config = TrainConfig(**config_dict)

"""
load data
"""

xnames = []
ynames = []

# mapping name to extension
found_exts = {}

# find which paths to load, by matching npy files with video files
for filename in os.listdir("data"):
    if not filename.startswith("."):
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
    vid = readvid("data/"+xnames[i])
    print(len(vid), "frames")
    xraw.append(vid)

if len(xraw) < 1 or len(yraw) < 1:
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
        if not ((pos_in_chord / chord_len < 0.2) or (pos_in_chord / chord_len > 0.7)):
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
IM_HEIGHT = 108
IM_WIDTH = 540
print("target height/width ratio:", IM_HEIGHT/IM_WIDTH)

xtrain, xval, xtest = [], [], []
ytrain, yval, ytest = [], [], []
print("Processing videos: ", end="")
for i in range(len(xs)):
    print(i, end=" ", flush=True)
    x = [cv.resize(im, dsize=(IM_WIDTH,IM_HEIGHT), interpolation=cv.INTER_AREA) for im in xs[i]]
    y = ys[i]
    if not args.nodisplay:
        showim(x[0], ms=400)

    # use last 15% of each video as test set
    testsplit = -int(0.15 * len(x))
    xtest += x[testsplit:]
    ytest += y[testsplit:]
    x = x[:testsplit]
    y = y[:testsplit]

    # use last 10% of video (excluding test set) as validation
    valsplit = -int(0.10 * len(x))
    xtrain += x[:valsplit]
    xval += x[valsplit:]
    ytrain += y[:valsplit]
    yval += y[valsplit:]

print()
# free up memory
del x, y, xs, ys

# convert to numpy
xtrain = np.array(xtrain)
xval = np.array(xval)
xtest = np.array(xtest)
ytrain = np.array(ytrain)
yval = np.array(yval)
ytest = np.array(ytest)

# shuffle train set
shuffle_inds = np.random.permutation(len(xtrain))
xtrain = xtrain[shuffle_inds]
ytrain = ytrain[shuffle_inds]

if not args.nodisplay:
    showvid(xtrain[:10], name="x", ms=100)

img_shape = xtrain[0].shape
print("img_shape", img_shape)
print(len(xtrain), "training images,", len(xval), "validation,", len(xtest), "test")

"""
make model
"""

def fret_accuracy():
    """
    round to nearest fret
    """
    acc = keras.metrics.Accuracy()
    def accuracy(y_true, y_pred):
        y_pred = tf.round(y_pred)
        return acc(y_true, y_pred)
    return accuracy


if not args.load:
    lossname = config.loss.lower()
    print("Using loss", lossname)
    categorical = False
    if lossname == "mse":
        loss = keras.losses.mean_squared_error
    elif lossname == "huber":
        loss = keras.losses.Huber(delta=1)
    elif lossname == "mae":
        loss = keras.losses.mean_absolute_error
    elif lossname == "sparsecategoricalcrossentropy":
        # regular scc from keras.losses doesnt work?
        loss = keras.losses.SparseCategoricalCrossentropy()
        categorical = True
    else:
        raise ValueError("No such loss '{}'".format(config.loss))

    model = make_model(config.model, img_shape, output_confidences=categorical)

    model.summary()

    if categorical:
        metrics = ["sparse_categorical_accuracy"]
    else:
        metrics = [fret_accuracy(), "mae"]

    model.compile(
        loss=loss,
        optimizer=Adam(config.lr),
        metrics=metrics,
    )


    """
    train model
    """

    def lr_sched(epoch, lr=None):
        if lr is None:
            if epoch % config.lr_sched_freq == 0:
                print("Decreasing learning rate to", lr)
            exp = epoch // config.lr_sched_freq
            lr = config.lr * (config.lr_sched_factor ** exp)
        elif epoch == 0:
            pass
        elif epoch % config.lr_sched_freq == 0:
            lr = lr * config.lr_sched_factor
            print("Decreasing learning rate to", lr)
        return lr

    os.makedirs("models/", exist_ok=True)
    callbacks = [
        History(),
        LearningRateScheduler(lr_sched),
        ModelCheckpoint("models/"+args.name+".hdf5", save_best_only=True, verbose=1, period=1),
        EarlyStopping(monitor='val_loss', verbose=1, patience=int(config.lr_sched_freq * 1.4))
    ]

    start = time.time()
    try:
        H = model.fit(
            xtrain,
            ytrain,
            validation_data=(xval, yval),
            batch_size=config.batchsize,
            epochs=config.epochs,
            verbose=1,
            callbacks=callbacks,
        )
    except KeyboardInterrupt:
        print("\nManual early stopping")
        H = callbacks[0]
    end = time.time()
    
    step = max(1, len(H.history['loss']) // 6)
    save_history(H, args.name, end-start, config, marker_step=step)


objs = {"accuracy": fret_accuracy()}

print("Loading model...")
model = keras.models.load_model("models/"+args.name+".hdf5", custom_objects=objs)

if args.load:
    model.summary()


"""
testing
"""

print("Evaluating on test set")
model.evaluate(xtest, ytest)

# on training set
num = 10
train_short = xtrain[:num]

trainpreds = model.predict(train_short)

scaleup = 2.0
vid = [cv.resize(i, dsize=(0,0), fx=scaleup, fy=scaleup, \
            interpolation=cv.INTER_LINEAR) for i in train_short]

annotate_vid(vid, trainpreds, ytrain[:num])
if not args.nodisplay:
    showvid(vid, name="train ims", ms=500)
writevid(vid, "stats/"+args.name+"/results_visualization_trainset")


# on test set
testpreds = model.predict(xtest)

vid = [cv.resize(i, dsize=(0,0), fx=scaleup, fy=scaleup, \
            interpolation=cv.INTER_LINEAR) for i in xtest]

annotate_vid(vid, testpreds, ytest)
if not args.nodisplay:
    showvid(vid, name="test set", ms=35)
writevid(vid, "stats/"+args.name+"/results_visualization_testset")
