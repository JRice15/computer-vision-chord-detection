import argparse
import json
import os
import time
import pprint

import keras
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

from src.cv_helpers import *
from src.models import make_model, fret_accuracy
from src.load_data import load_one, load_all_data

def test_im_model(name, xtest, ytest, xtrain_short=None, ytrain_short=None, nodisplay=False, 
        summary=False):
    """
    test the image model on test set data
    args:
        xtest, ytest: main test vids
        xtrain_short, ytrain_short: shuffled short selections from xtrain
        summary: whether to show summary
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
    print("Evaluating on test set")
    print(len(xtest), "testing images")
    results = model.evaluate(xtest, ytest, verbose=1)
    with open("stats/"+name+"/stats.txt", "a") as f:
        f.write("\nTest results:\n")
        for i,metric in enumerate(model.metrics_names):
            print(" ", metric+":", results[i])
            f.write(metric+": "+str(results[i])+"\n")

    scaleup = 2.0

    # on training set, if available
    if xtrain_short is not None:
        print("Generating video on train set predictions")
        trainpreds = model.predict(xtrain_short, verbose=1)

        vid = [cv.resize(i, dsize=(0,0), fx=scaleup, fy=scaleup, \
                    interpolation=cv.INTER_LINEAR) for i in train_short]

        annotate_vid(vid, trainpreds, ytrain_short, categorical)
        if not nodisplay:
            showvid(vid, name="train ims", ms=300)
        writevid(vid, "stats/"+name+"/results_visualization_trainset")

    # on test set
    print("Generating video on test set predictions")
    testpreds = model.predict(xtest, verbose=1)

    vid = [cv.resize(i, dsize=(0,0), fx=scaleup, fy=scaleup, \
                interpolation=cv.INTER_LINEAR) for i in xtest]

    annotate_vid(vid, testpreds, ytest, categorical)
    if not nodisplay:
        showvid(vid, name="test set", ms=35)
    writevid(vid, "stats/"+name+"/results_visualization_testset")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--name",required=True,help="name of model to load")
    parser.add_argument("--nodisplay",action="store_true")
    parser.usage = parser.format_help()
    args = parser.parse_args()

    data = load_all_data("data/inference_model_train", num_splits=0, split_amount=0.1,
            display=(not args.nodisplay))
    xtest, _, _, ytest, _, _ = data

    test_im_model(args.name, xtest, ytest, 
        nodisplay=args.nodisplay, summary=True)
