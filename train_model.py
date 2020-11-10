import argparse
import json
import os

import keras
import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.callbacks import (History, LearningRateScheduler, ModelCheckpoint,
                             ReduceLROnPlateau)
from keras.models import Model
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split

from cv_helpers import *
from models import make_model

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

with open("model_config.json", "r") as f:
    config_dict = json.load(f)

config = TrainConfig(**config_dict)

"""
load data
"""

yraw = []
xraw = []

for name in os.listdir("data"):
    if not name.startswith("."):
        path = "data/"+name
        if split_path(name)[-1] == ".npy":
            yraw.append(np.load(path))
        else:
            vid = readvid(path)
            print(len(vid), "frames")
            xraw.append(vid)

if len(xraw) < 1 or len(yraw) < 1:
    print("No data in the 'data' directory")
    exit()

xs = []
ys = []

def str_to_chord(s):
    return np.array(list(s), dtype=int)

for i in range(len(yraw)):
    start = int(yraw[i][0][0]) # first chord, its index
    xs.append(xraw[i][start:])
    thisy = []
    # frame index, relative to video
    vid_ind = start
    # index of next upcoming chord in y[i]
    y_ind = 0
    while vid_ind < len(xraw[i]):
        # if the frameindex is 
        if y_ind < len(yraw[i]) and int(yraw[i][y_ind][0]) == vid_ind:
            chord = str_to_chord(yraw[i][y_ind][1])
            y_ind += 1
        thisy.append(chord)
        vid_ind += 1
    ys.append(thisy)

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
# TODO when loading multiple vids, make all the same shape. and downsample?

x = []
y = []
for i in range(len(xs)):
    x += xs[i]
    y += ys[i]

resize_factor = 0.4
x = [cv.resize(i, dsize=(0,0), fx=resize_factor, fy=resize_factor, interpolation=cv.INTER_AREA) for i in x]

img_shape = x[0].shape

showim(x[0], ms=1000)
print("img_shape", img_shape)

split_idx = len(x) // 5
xtest = x[-split_idx:]
x = x[:-split_idx]
ytest = y[-split_idx:]
y = y[:-split_idx]

xtrain, xval, ytrain, yval = train_test_split(x, y, test_size=0.15, random_state=3, shuffle=True)

print(len(xtrain), "training images,", len(xval), "validation,", len(xtest), "test")

"""
make model
"""
model = make_model(config.model, img_shape)

model.summary()

def fret_accuracy(y_true, y_pred):
    """
    average number of correctly predicted frets
    """
    y_pred = tf.round(y_pred)
    corrects = (y_true == y_pred)
    return K.mean(tf.cast(corrects, tf.int16))

model.compile(
    loss="mse",
    optimizer=Adam(config.lr),
    metrics=[fret_accuracy, "mae"])


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


if args.load:
    print("Loading weights...")
    model.load_weights("models/"+args.name+".hdf5")

else:
    os.makedirs("models/", exist_ok=True)
    callbacks = [
        History(),
        LearningRateScheduler(lr_sched),
        ModelCheckpoint("models/"+args.name+".hdf5", save_best_only=True, verbose=1, period=1)
    ]

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


"""
testing
"""

print("Evaluating on test set")
model.evaluate(np.array(xtest), np.array(ytest))

testpreds = model.predict(np.array(xtest))

vid = [cv.resize(i, dsize=(0,0), fx=1/resize_factor, fy=1/resize_factor, \
            interpolation=cv.INTER_LINEAR) for i in xtest]

annotate_vid(vid, testpreds, ytest)

showvid(vid, ms=35)
