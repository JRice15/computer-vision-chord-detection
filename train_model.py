import argparse
import json
import os
import time

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
args = parser.parse_args()

class TrainConfig:

    def __init__(self, epochs, model, batchsize, lr, lr_sched_freq, lr_sched_factor):
        self.epochs = epochs
        self.model = model
        self.batchsize = batchsize
        self.lr = lr
        self.lr_sched_freq = lr_sched_freq
        self.lr_sched_factor = lr_sched_factor
    
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

"""
preprocessing
"""
# TODO when loading multiple vids, make all the same shape

x = []
y = []
for i in range(len(xs)):
    x += xs[i]
    y += ys[i]

# showvid(x[-300:], name="x", ms=100)

resize_factor = 0.5
x = [cv.resize(i, dsize=(0,0), fx=resize_factor, fy=resize_factor, interpolation=cv.INTER_AREA) for i in x]

img_shape = x[0].shape

showim(x[0], ms=1000)
print("img_shape", img_shape)

split_idx = len(x) // 5
xtest = x[-split_idx:]
x = x[:-split_idx]
ytest = y[-split_idx:]
y = y[:-split_idx]

xtrain, xval, ytrain, yval = train_test_split(x, y, test_size=0.15, random_state=4, shuffle=True)

print(len(xtrain), "training images,", len(xval), "validation,", len(xtest), "test")

"""
make model
"""

class FretAccuracy(keras.metrics.Accuracy):
    """
    custom metric that rounds predictions to closest integer
    """

    def update_state(self, y_true, y_pred):
        y_pred = tf.round(y_pred)
        return super().update_state(y_true, y_pred)


if not args.load:
    model = make_model(config.model, img_shape)

    model.summary()

    model.compile(
        loss="mse",
        optimizer=Adam(config.lr),
        metrics=[FretAccuracy(), "mae"])


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
            np.array(xtrain),
            np.array(ytrain),
            validation_data=(np.array(xval), np.array(yval)),
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


objs = {"FretAccuracy": FretAccuracy}

print("Loading model...")
model = keras.models.load_model("models/"+args.name+".hdf5", custom_objects=objs)

if args.load:
    model.summary()


"""
testing
"""

print("Evaluating on test set")
model.evaluate(np.array(xtest), np.array(ytest))

# on training set
num = 10
train_short = np.array(xtrain[:num])

trainpreds = model.predict(train_short)

vid = [cv.resize(i, dsize=(0,0), fx=1/resize_factor, fy=1/resize_factor, \
            interpolation=cv.INTER_LINEAR) for i in train_short]

annotate_vid(vid, trainpreds, ytrain[:num])
showvid(vid, name="train ims", ms=500)
writevid(vid, "stats/"+args.name+"/results_visualization_trainset")


# on test set
testpreds = model.predict(np.array(xtest))

vid = [cv.resize(i, dsize=(0,0), fx=1/resize_factor, fy=1/resize_factor, \
            interpolation=cv.INTER_LINEAR) for i in xtest]

annotate_vid(vid, testpreds, ytest)
showvid(vid, name="test set", ms=35)
writevid(vid, "stats/"+args.name+"/results_visualization_testset")
