"""
This file is to train and test the model that takes in multiple predictions from
the large single-frame model, and learns to improve its predictions by considering
the neighbors of each prediction

The model is fairly small
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
from sklearn.model_selection import train_test_split

from src.cv_helpers import *
from src.models import fret_accuracy, make_inference_model, MyConv1DTranspose
from src.save_stats import save_history
from src.load_data import load_data

parser = argparse.ArgumentParser()
parser.add_argument("--name",required=True)
parser.add_argument("--load",action="store_true")
parser.add_argument("--nodisplay",action="store_true")
parser.add_argument("--test",action="store_true",help="load a small portion of the data for a quick test run")
parser.add_argument("--repredict",action="store_true",help="re-predict; run the imagemodel on the training data again, to generate update predictions")
parser.usage = parser.format_help()
args = parser.parse_args()

modelname = args.name + "_inference"

class TrainConfig:

    def __init__(self):
        self.epochs = 100
        self.batches_per_epoch = 200
        self.batchsize = 32
        self.lr = 0.001
        self.lr_sched = [10, 30, 50, 70, 90]
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

ytrain, yval, ytest = load_data(not args.nodisplay, args.test, y_only=True)

class PredLoader():

    def __init__(self, name):
        self.name = name
        # we delay model and data loading until we need it, cuz its pretty big
        self.img_model = None
        self.data = None

    def get_data_by_type(self, typ):
        if typ == "train":
            return self.data[0]
        elif typ == "val":
            return self.data[1]
        elif typ == "test":
            return self.data[2]
        else:
            raise ValueError("bad load type")

    def load(self, typ):
        print("Loading", typ, "imagemodel predictions")
        path = "preds/"+typ+"_"+self.name+".npy"
        if os.path.exists(path) and not args.repredict:
            preds = np.load(path)
        else:
            preds = self.make_predictions(typ, path)
        return preds

    def make_predictions(self, typ, savepath):
        # load data/model, if not loaded already
        if self.img_model is None:

            print("Loading video data")
            data = load_data(not args.nodisplay, args.test)
            xtrain, xval, xtest, ytrain, yval, ytest = data
            self.data = (xtrain, xval, xtest)

            print("Loading image model...")
            objs = {"accuracy": fret_accuracy()}
            self.img_model = keras.models.load_model("models/"+args.name+".hdf5", custom_objects=objs)

        x = self.get_data_by_type(typ)
        # make predictions
        preds = self.img_model.predict(x, verbose=1)
        print("Saving", savepath)
        np.save(savepath, preds)
        return preds

# load predictions, or run predictions if there is unpredicted data
os.makedirs("preds", exist_ok=True)

loader = PredLoader(args.name)

xpredtrain = loader.load("train")
xpredval = loader.load("val")
xpredtest = loader.load("test")

if len(xpredtrain) != len(ytrain) or len(xpredval) != len(yval) or \
        len(xpredtest) != len(ytest):
    raise ValueError("Loaded predictions don't match y data. Run again with '--repredict'")

print(len(xpredtrain), "training,", len(xpredval), "validation,", len(xpredtest), "testing preds")

# free memory; this deletes the model and data
del loader

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
            
def batch_data(x, y, step=50):
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

xpredval, yval = batch_data(xpredval, yval)
xpredtest, ytest = batch_data(xpredtest, ytest)


"""
training
"""
if not args.load:

    pred_shape = xpredtrain[0].shape
    input_shape = (config.input_length,) + pred_shape

    categorical = (len(pred_shape) > 1)

    model = make_inference_model(input_shape, categorical)
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
        EarlyStopping(verbose=1, patience=(config.epochs//4))
    ]

    start = time.time()
    try:
        H = model.fit(
            train_gen(),
            validation_data=(xpredval, yval),
            batch_size=config.batchsize,
            epochs=config.epochs,
            verbose=1,
            steps_per_epoch=config.batches_per_epoch,
            callbacks=callbacks,
        )
    except KeyboardInterrupt:
        print("\nManual early stopping")
        H = callbacks[0]
    end = time.time()
    
    step = max(1, len(H.history['loss']) // 6)
    save_history(H, modelname, end-start, config, marker_step=step)



print("Loading model...")
objs = {"accuracy": fret_accuracy(), "MyConv1DTranspose": MyConv1DTranspose}
model = keras.models.load_model("models/"+modelname+".hdf5", custom_objects=objs)


if args.load:
    # if we are just loading and have not trained
    model.summary()

    # if (batchsize, guitarstringindex, probabilities) then categorical, else 
    #   (batchsize, stringpred) is regression-type
    shape = model.get_output_shape_at(-1)
    if len(shape) > 2:
        categorical = True
    else:
        categorical = False


"""
testing
"""

print("Evaluating on test set")
results = model.evaluate(xpredtest, ytest)
with open("stats/"+args.name+"/stats.txt", "a") as f:
    f.write("\nInference Test results:\n")
    for i,name in enumerate(model.metrics_names):
        print(" ", name+":", results[i])
        f.write(name+": "+str(results[i])+"\n")


# on training set
train_short = xpredtrain[:config.input_length]

trainpreds = model.predict(train_short)

scaleup = 2.0
vid = [cv.resize(i, dsize=(0,0), fx=scaleup, fy=scaleup, \
            interpolation=cv.INTER_LINEAR) for i in train_short]

annotate_vid(vid, trainpreds, ytrain[:config.input_length], categorical)
if not args.nodisplay:
    showvid(vid, name="train ims", ms=500)
writevid(vid, "stats/"+args.name+"/results_visualization_trainset")


# on test set
testpreds = model.predict(xtest)

vid = [cv.resize(i, dsize=(0,0), fx=scaleup, fy=scaleup, \
            interpolation=cv.INTER_LINEAR) for i in xtest]

annotate_vid(vid, testpreds, ytest, categorical)
if not args.nodisplay:
    showvid(vid, name="test set", ms=35)
writevid(vid, "stats/"+args.name+"/results_visualization_testset")

