import argparse
import json
import os
import time
import pprint

import keras
import numpy as np
import tensorflow as tf

from src.cv_helpers import *
from src.models import make_inference_model, MyConv1DTranspose, fret_accuracy
from src.load_data import load_one, load_all_data


def test_inference_model(im_name, inf_name, xpredtest, ytest, xpredtrain=None, 
        ytrain=None, summary=False, display=False):
    """
    args:
        im_name: name of image model
        inf_name: name of inference model
    """
    print("Loading model...")
    objs = {"accuracy": fret_accuracy(), "MyConv1DTranspose": MyConv1DTranspose}
    model = keras.models.load_model("models/"+inf_name+".hdf5", custom_objects=objs)


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
    input_length = xpredtest[0].shape[0]
    print("input_length:", input_length)


    """
    testing
    """

    print("Evaluating on test set")
    results = model.evaluate(xpredtest, ytest)
    with open("stats/"+im_name+"/stats.txt", "a") as f:
        f.write("\nInference Test results:\n")
        for i,name in enumerate(model.metrics_names):
            print(" ", name+":", results[i])
            f.write(name+": "+str(results[i])+"\n")

    scaleup = 2.0

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


    # # on test set
    # testpreds = model.predict(xpredtest)

    # vid = [cv.resize(i, dsize=(0,0), fx=scaleup, fy=scaleup, \
    #             interpolation=cv.INTER_LINEAR) for i in xpredtest]

    # annotate_vid(vid, testpreds, ytest, categorical)
    # if display:
    #     showvid(vid, name="test set", ms=35)
    # writevid(vid, "stats/"+inf_name+"/results_visualization_testset")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--name",required=True,help="name of model to load")
    parser.add_argument("--nodisplay",action="store_true")
    parser.usage = parser.format_help()
    args = parser.parse_args()

    PRED_DIR = "data/image_preds/"
    xpredtest = np.load(PRED_DIR+"train_"+args.name+".npy")
    ytest, _, _ = load_all_data("data/inference_model_test", num_splits=0,
        display=(not args.nodisplay), y_only=True, no_transitions=False)

    test_inference_model(args.name+"_inference", args.name, xpredtest, ytest,
        summary=True, display=(not args.nodisplay))
