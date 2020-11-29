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
from src.load_data import load_all_data, load_one


def test_im_model(name, xtest, ytest, xtrain=None, ytrain=None, nodisplay=False, 
        summary=False, categorical=False):
    """
    test the image model on test set data
    """
    print("Loading model...")
    objs = {"accuracy": fret_accuracy()}
    model = keras.models.load_model("models/"+name+".hdf5", custom_objects=objs)


    if summary:
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

    print(model.callbacks)
    a = [i for i in model.callbacks if isinstance(i, keras.callbacks.ProgbarLogger)][0]
    print(a)

    print("Evaluating on test set")
    results = model.evaluate(xtest, ytest)
    with open("stats/"+args.name+"/stats.txt", "a") as f:
        f.write("\nTest results:\n")
        for i,name in enumerate(model.metrics_names):
            print(" ", name+":", results[i])
            f.write(name+": "+str(results[i])+"\n")


    scaleup = 2.0

    if xtrain is not None:
        # on training set
        num = 10
        train_short = xtrain[:num]

        trainpreds = model.predict(train_short)

        vid = [cv.resize(i, dsize=(0,0), fx=scaleup, fy=scaleup, \
                    interpolation=cv.INTER_LINEAR) for i in train_short]

        annotate_vid(vid, trainpreds, ytrain[:num], categorical)
        if not args.nodisplay:
            showvid(vid, name="train ims", ms=500)
        writevid(vid, "stats/"+name+"/results_visualization_trainset")


    # on test set
    testpreds = model.predict(xtest)

    vid = [cv.resize(i, dsize=(0,0), fx=scaleup, fy=scaleup, \
                interpolation=cv.INTER_LINEAR) for i in xtest]

    annotate_vid(vid, testpreds, ytest, categorical)
    if not args.nodisplay:
        showvid(vid, name="test set", ms=35)
    writevid(vid, "stats/"+name+"/results_visualization_testset")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--name",required=True,help="name of model to load")
    parser.add_argument("--file",required=True,help="path to vid file (npy file should be in the same directory, and have the same name other than the extension")
    parser.add_argument("--nodisplay",action="store_true")
    parser.usage = parser.format_help()
    args = parser.parse_args()

    directory, file, ext = split_path(args.file)
    yfile = directory + "/" + file + ".npy"
    x, y = load_one(args.file, yfile, not args.nodisplay)

    test_im_model(args.name, x, y, nodisplay=args.nodisplay, summary=True, categorical=None)
