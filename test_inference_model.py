import argparse
import json
import os
import pprint
import time

import keras
import numpy as np
import tensorflow as tf

from src.cv_helpers import *
from src.image_models import fret_accuracy
from src.inference_models import (INPUT_LEN, MyConv1DTranspose,
                                  group_sequences, make_inference_model)
from src.load_data import load_all_data, load_one
from src.load_preds import load_predictions


def test_inference_model(im_name, inf_name, xpredtrain=None, 
        ytrain=None, summary=False, display=False, noimagemodel=False, makevid=False):
    """
    args:
        im_name: name of image model
        inf_name: name of inference model
    """
    print("Loading models...")
    objs = {"accuracy": fret_accuracy(), "MyConv1DTranspose": MyConv1DTranspose}
    inf_model = keras.models.load_model("models/"+inf_name+".hdf5", custom_objects=objs)
    if not noimagemodel:
        img_model = keras.models.load_model("models/"+im_name+".hdf5", custom_objects=objs)

    if summary:
        inf_model.summary()

    """
    load data
    """    
    preddata = load_predictions(im_name, display=display)
    _, _, xpredtest, categorical = preddata
    
    if noimagemodel and not makevid:
        data = load_all_data("data/inference_model_test", num_splits=0,
            display=display, no_transitions=False, y_only=True)
        ytest, _, _ = data
    else:
        data = load_all_data("data/inference_model_test", num_splits=0,
            display=display, no_transitions=False)
        xtest, _, _, ytest, _, _ = data

    """
    testing
    """

    xpredtestgrouped, ytestgrouped = group_sequences(xpredtest, ytest, step=1)

    print("Evaluating inference model on test set")
    print(len(xpredtestgrouped), "test examples")
    results = inf_model.evaluate(xpredtestgrouped, ytestgrouped)
    with open("stats/"+inf_name+"/stats.txt", "a") as f:
        f.write("\nInference Test results:\n")
        for i,name in enumerate(inf_model.metrics_names):
            print(" ", name+":", results[i])
            f.write(name+": "+str(results[i])+"\n")

    if not noimagemodel:
        print("Evaluating image model on test set")
        print(len(xtest), "test examples")
        results = img_model.evaluate(xtest, ytest)
        with open("stats/"+img_model+"/stats.txt", "a") as f:
            f.write("\nInference Test results:\n")
            for i,name in enumerate(img_model.metrics_names):
                print(" ", str(name)+":", results[i])
                f.write(str(name)+": "+str(results[i])+"\n")

    scaleup = 2.0

    if makevid:
        # on training set
        # if xpredtrain is not None:
        #     train_short = xpredtrain[:input_length]

        #     trainpreds = model.predict(train_short)

        #     vid = [cv.resize(i, dsize=(0,0), fx=scaleup, fy=scaleup, \
        #                 interpolation=cv.INTER_LINEAR) for i in train_short]

        #     annotate_vid(vid, trainpreds, ytrain[:input_length], categorical)
        #     if display:
        #         showvid(vid, name="train ims", ms=500)
        #     writevid(vid, "stats/"+inf_name+"/results_visualization_trainset")


        # on test set
        print("Generating video on test set predictions")
        xpredtest2, ytest2 = group_sequences(xpredtest, ytest, step=INPUT_LEN)

        numframes = 3000 // INPUT_LEN
        testpreds = inf_model.predict(xpredtest2[:numframes], verbose=1)

        vid = [cv.resize(i, dsize=(0,0), fx=scaleup, fy=scaleup, \
                    interpolation=cv.INTER_LINEAR) for i in xtest[:numframes*INPUT_LEN]]

        testpreds = np.concatenate(list(testpreds), axis=0)
        ytest2 = np.concatenate(list(ytest2[:numframes]), axis=0)
        annotate_vid(vid, testpreds, ytest2, categorical)
        if display:
            showvid(vid, name="test set", ms=35)
        writevid(vid, "stats/"+inf_name+"/results_visualization_testset")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--name",required=True,help="name of model to load")
    parser.add_argument("--nodisplay",action="store_true")
    parser.add_argument("--noimagemodel",action="store_true",help="only run the inference model")
    parser.usage = parser.format_help()
    args = parser.parse_args()

    test_inference_model(args.name, args.name+"_inference", 
        summary=True, display=(not args.nodisplay), noimagemodel=args.noimagemodel,
        makevid=True)
