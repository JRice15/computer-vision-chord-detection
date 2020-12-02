import argparse
import json
import os
import time
import pprint

import keras
import numpy as np
import tensorflow as tf

from src.cv_helpers import *
from src.models import make_model, fret_accuracy
from src.load_data import load_one, load_all_data

def test_im_model(name, xtrain_short=None, ytrain_short=None, nodisplay=False, 
        summary=False, do_test=False):
    """
    test the image model on test set data
    args:
        name: name of model to load
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
    data = load_all_data("data/inference_model_train", num_splits=0, 
                display=(not args.nodisplay), do_test=do_test)
    xtest, _, _, ytest, _, _ = data

    print("Evaluating on test set w/ no transitions")
    print(len(xtest), "testing images")
    results = model.evaluate(xtest, ytest, verbose=1)
    with open("stats/"+name+"/stats.txt", "a") as f:
        f.write("\nTest results (no transitions):\n")
        for i,metric in enumerate(model.metrics_names):
            print(" ", metric+":", results[i])
            f.write(metric+": "+str(results[i])+"\n")

    data = load_all_data("data/inference_model_train", num_splits=0, 
                display=(not args.nodisplay), do_test=do_test, no_transitions=False)
    xtest, _, _, ytest, _, _ = data

    print("Evaluating on test set w/ transitions")
    print(len(xtest), "testing images")
    results = model.evaluate(xtest, ytest, verbose=1)
    with open("stats/"+name+"/stats.txt", "a") as f:
        f.write("\nTest results (with transitions):\n")
        for i,metric in enumerate(model.metrics_names):
            print(" ", metric+":", results[i])
            f.write(metric+": "+str(results[i])+"\n")

    scaleup = 2.0

    # on training set, if available
    if xtrain_short is not None:
        print("Generating video on train set predictions")
        trainpreds = model.predict(xtrain_short, verbose=1)

        vid = [cv.resize(i, dsize=(0,0), fx=scaleup, fy=scaleup, \
                    interpolation=cv.INTER_LINEAR) for i in xtrain_short]

        annotate_vid(vid, trainpreds, ytrain_short, categorical)
        if not nodisplay:
            showvid(vid, name="train ims", ms=300)
        writevid(vid, "stats/"+name+"/results_visualization_trainset")

    # on test set
    print("Generating video on test set predictions")
    numframes = 1000
    testpreds = model.predict(xtest[:numframes], verbose=1)

    vid = [cv.resize(i, dsize=(0,0), fx=scaleup, fy=scaleup, \
                interpolation=cv.INTER_LINEAR) for i in xtest[:numframes]]

    annotate_vid(vid, testpreds, ytest[:numframes], categorical)
    if not nodisplay:
        showvid(vid, name="test set", ms=35)
    writevid(vid, "stats/"+name+"/results_visualization_testset")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--name",required=True,help="name of model to load")
    parser.add_argument("--nodisplay",action="store_true")
    parser.usage = parser.format_help()
    args = parser.parse_args()

    test_im_model(args.name, nodisplay=args.nodisplay, summary=True)
