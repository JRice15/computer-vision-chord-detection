"""
This file is to train and test the model that takes in multiple predictions from
the large single-frame model, and learns to improve its predictions by considering
the neighbors of each prediction

The model is fairly small so you can train without a GPU
"""

import argparse
import json
import logging
import os
import random
import pprint
import time

import keras
import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.callbacks import (EarlyStopping, History, LearningRateScheduler,
                             ModelCheckpoint, ReduceLROnPlateau)
from keras import layers
from keras.models import Model
from keras.optimizers import Adam

from src.cv_helpers import *
from src.models import fret_accuracy, make_inference_model, MyConv1DTranspose, get_output_shape
from src.save_stats import save_history
from src.load_data import load_all_data
from src.load_preds import load_predictions
from test_inference_model import test_inference_model

parser = argparse.ArgumentParser()
parser.add_argument("--name",required=True)
parser.add_argument("--load",action="store_true")
parser.add_argument("--nodisplay",action="store_true")
parser.add_argument("--test",action="store_true",help="load a small portion of the data for a quick test run")
parser.add_argument("--repredict",action="store_true",help="re-predict; run the imagemodel on the training data again, to generate updated predictions")
parser.usage = parser.format_help()
args = parser.parse_args()

modelname = args.name + "_inference"

class TrainConfig:

    def __init__(self):
        self.epochs = 50
        self.batchsize = 32
        self.lr = 0.001
        self.lr_sched = [4, 12, 24, 36, 44]
        self.lr_sched_factor = 0.2
        self.input_length = 64
        pprint.pprint(vars(self))
    
    def __str__(self):
        return str(vars(self))
    
    def write_to_file(self,filename):
        with open(filename, "a") as f:
            f.write("\n" + str(self) + "\n\n")

config = TrainConfig()

"""
load data
"""

ytrain, yval, _ = load_all_data("data/inference_model_train", num_splits=1,
    display=(not args.nodisplay), do_test=args.test, y_only=True, no_transitions=False)

ytest, _, _ = load_all_data("data/inference_model_test", num_splits=0,
    display=(not args.nodisplay), do_test=args.test, y_only=True, no_transitions=False)


data = load_predictions(args.name, display=(not args.nodisplay), repredict=args.repredict)
xpredtrain, xpredval, xpredtest, categorical = data

if len(xpredtrain) != len(ytrain) or len(xpredval) != len(yval) or \
        len(xpredtest) != len(ytest):
    raise ValueError("Loaded predictions don't match target y data. Run again with '--repredict'")

print(len(xpredtrain), "training,", len(xpredval), "validation,", len(xpredtest), "testing preds")


"""
preparing data
"""

def train_gen():
    """
    generate a batch of X,Y
    """
    X = np.empty((config.batchsize,)+input_shape)
    Y = np.empty((config.batchsize,config.input_length)+ytrain[0].shape)
    while True:
        for i in range(config.batchsize):
            j = np.random.randint(len(xpredtrain)-config.input_length)
            flip = random.choice([True, False])
            if flip:
                X[i] = xpredtrain[j+config.input_length:j:-1]
                Y[i] = ytrain[j+config.input_length:j:-1]
            else:
                X[i] = xpredtrain[j:j+config.input_length]
                Y[i] = ytrain[j:j+config.input_length]
        # noise to prevent overfitting
        X += np.random.normal(0.0, 0.10, size=X.shape)
        yield X, Y
            
def group_sequences(x, y, step=50):
    """
    split data into input_length sized pieces
    """
    inptlen = config.input_length
    X = np.empty((len(x)//step, inptlen)+x.shape[1:])
    Y = np.empty((len(y)//step, inptlen)+y.shape[1:])
    for batch_ind,ind in enumerate(range(inptlen, len(x), step)):
        X[batch_ind] = x[ind-inptlen:ind]
        Y[batch_ind] = y[ind-inptlen:ind]
    return X, Y

xpredval, yval = group_sequences(xpredval, yval)
xpredtest, ytest = group_sequences(xpredtest, ytest)


"""
training
"""

pred_shape = xpredtrain[0].shape
input_shape = (config.input_length,) + pred_shape
output_shape = (config.input_length,) + get_output_shape(categorical=categorical)
print("input, output shapes:", input_shape, output_shape)

model = make_inference_model(input_shape, output_shape, categorical=categorical)
model.summary()

if categorical:
    loss = "sparse_categorical_crossentropy"
    metrics = ["sparse_categorical_accuracy"]
else:
    loss = "mse"
    metrics = [fret_accuracy(), "mae"]

model.compile(
    loss=loss,
    optimizer=Adam(config.lr),
    metrics=metrics,
)

"""
train model
"""

def lr_sched(epoch, lr):
    if epoch in config.lr_sched:
        lr *= config.lr_sched_factor
        print("Decreasing learning rate to", lr)
    return lr

os.makedirs("models/", exist_ok=True)
callbacks = [
    History(),
    LearningRateScheduler(lr_sched),
    ModelCheckpoint("models/"+modelname+".hdf5", save_best_only=True, verbose=1, period=1),
    EarlyStopping(verbose=1, patience=(config.epochs//3))
]

start = time.time()
try:
    H = model.fit(
        train_gen(),
        validation_data=(xpredval, yval),
        batch_size=config.batchsize,
        epochs=config.epochs,
        verbose=1,
        steps_per_epoch=(len(xpredtrain)//(2*config.batchsize)), # half of dataset per epoch
        callbacks=callbacks,
    )
except KeyboardInterrupt:
    print("\nManual early stopping")
    H = callbacks[0]
end = time.time()

step = max(1, len(H.history['loss']) // 6)
save_history(H, modelname, end-start, config, marker_step=step)


test_inference_model(args.name, modelname, xpredtest, ytest, display=(not args.nodisplay))
