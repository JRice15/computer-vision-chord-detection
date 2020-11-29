"""
This file is to train the model that takes in individual video frames, and outputs
predicted probabilities of what fret is held down for each string

A present, the models used are pretty large, and should probably be trained
on a GPU
"""

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

from src.cv_helpers import *
from src.models import make_model, fret_accuracy
from src.save_stats import save_history
from src.load_data import load_all_data
from test_image_model import test_im_model


parser = argparse.ArgumentParser()
parser.add_argument("--name",required=True,help="name to save the model under")
parser.add_argument("--nodisplay",action="store_true")
parser.add_argument("--test",action="store_true",help="load a small portion of the data for a quick test run")
parser.usage = parser.format_help()
args = parser.parse_args()

if args.test:
    args.name += "_test"

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

train data is loaded from data/image_model_train, while test data is from data/inference_model_train
"""
data = load_all_data("data/image_model_train", num_splits=1, 
            display=(not args.nodisplay), do_test=args.test)
xtrain, xval, _, ytrain, yval, _ = data

# shuffle train set
shuffle_inds = np.random.permutation(len(xtrain))
xtrain = xtrain[shuffle_inds]
ytrain = ytrain[shuffle_inds]

print(len(xtrain), "training images,", len(xval), "validation,")

img_shape = xtrain[0].shape

"""
make model
"""


# get the loss function
lossname = config.loss.lower()
print("Using loss", lossname)
categorical = False # whether output is regression or categorization
if lossname == "mse":
    loss = keras.losses.mean_squared_error
elif lossname == "huber":
    loss = keras.losses.Huber(delta=1)
elif lossname == "mae":
    loss = keras.losses.mean_absolute_error
elif lossname == "sparsecategoricalcrossentropy":
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
    EarlyStopping(monitor='val_loss', verbose=1, patience=int(config.lr_sched_freq * 1.5))
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


"""
testing
"""

data = load_all_data("data/inference_model_train", num_splits=0, 
            display=(not args.nodisplay), do_test=args.test)
xtest, _, _, ytest, _, _ = data

test_im_model(args.name, xtest, ytest, xtrain=xtrain, ytrain=ytrain, nodisplay=args.nodisplay, 
    summary=False, categorical=categorical)
