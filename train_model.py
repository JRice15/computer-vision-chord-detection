import json
import os
import argparse

import keras
import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.models import Model
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from keras.callbacks import History, ReduceLROnPlateau, ModelCheckpoint

from models import make_model
from cv_helpers import *

parser = argparse.ArgumentParser()
parser.add_argument("--name",required=True)
args = parser.parse_args()

class TrainConfig:

    def __init__(self, epochs, model, batchsize, lr):
        self.epochs = epochs
        self.model = model
        self.batchsize = batchsize
        self.lr = lr
    
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
            xraw.append(readvid(path))

print(len(xraw[0]))

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
    while vid_ind < len(xraw[i]) + start:
        # if the frameindex is 
        if y_ind < len(yraw[i]) and int(yraw[i][y_ind][0]) == vid_ind:
            chord = str_to_chord(yraw[i][y_ind][1])
            y_ind += 1
        thisy.append(chord)
        vid_ind += 1
    ys.append(thisy)

assert len(xs) == len(ys)
for i in range(len(xs)):
    assert len(xs[i]) == len(ys[i])

"""
preprocessing
"""
# TODO when loading multiple vids, make all the same shape. and downsample?

x = []
y = []
for i in range(len(xs)):
    x += xs[i]
    y += ys[i]

img_shape = x[0].shape

print("img_shape", img_shape)



"""
make model
"""

model = make_model(config.model, img_shape)

model.compile(
    loss="mse",
    optimizer=Adam(config.lr),
    metrics=['accuracy'])

os.makedirs("models/")
callbacks = [
    History(),
    ReduceLROnPlateau(factor=0.1, patience=10, verbose=1),
    ModelCheckpoint("models/"+args.name+".hdf5", save_best_only=True, verbose=1, period=5)
]

try:
    H = model.fit(
        x[0], y[0],
        batch_size=config.batchsize,
        epochs=config.epochs,
        verbose=1,
        callbacks=callbacks,
        validation_split=0.2,
        shuffle=True
    )
except KeyboardInterrupt:
    print("\nManual early stopping")
    H = callbacks[0]


