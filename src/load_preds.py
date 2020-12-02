import numpy as np
import os
import sys
import argparse
sys.path.append(".")
from src.cv_helpers import *
from src.models import fret_accuracy, MyConv1DTranspose
from src.load_data import load_all_data
import keras
from keras import Model

PRED_DIR = "data/image_preds/"
os.makedirs(PRED_DIR, exist_ok=True)

def same_shape(shape1, shape2):
    return len(shape1) == len(shape2) and \
        all([shape1[i] == shape2[i] for i in range(len(shape1))])

def get_pred_model(modelname):
    """
    returns:
        model (keras model), categorical (bool)
    """
    print("Loading image model...")
    objs = {"accuracy": fret_accuracy()}
    img_model = keras.models.load_model("models/"+modelname+".hdf5", custom_objects=objs)

    inpt = img_model.input
    out_layer = img_model.layers[-8]
    output = out_layer.output
    categorical = (len(img_model.get_output_shape_at(-1)) > 2)

    print("pred input tensor:", inpt)
    print("pred output layer:", out_layer)
    print("pred output tensor:", output)
    print("categorical output:", categorical)
    assert same_shape(output.shape, (None, 256))
    assert "re_lu" in output.name

    new_model = Model(inpt, output)
    new_model.compile()
    return new_model, categorical


class PredLoader():
    """
    class to load predictions, or re-run predictions
    """

    def __init__(self, name, display, repredict=False):
        self.name = name
        self.display = display
        self.repredict = repredict
        # we delay model and data loading until we need it, cuz its pretty big
        self.img_model = None
        self.train_data = None
        self.val_data = None
        self.test_data = None
        self.categorical = None

    def get_data_by_type(self, typ):
        if typ == "train":
            return self.train_data
        elif typ == "val":
            return self.val_data
        elif typ == "test":
            return self.test_data
        else:
            raise ValueError("bad load type")

    def load(self, typ):
        print("Loading", typ, "imagemodel predictions")
        path = PRED_DIR+typ+"_"+self.name+".npz"
        if os.path.exists(path) and not self.repredict:
            data = np.load(path)
            preds = data['preds']
            self.categorical = data['categorical']
        else:
            preds = self.make_predictions(typ, path)
        return preds, self.categorical

    def make_predictions(self, typ, savepath):
        # load data/model, if not loaded already
        if self.img_model is None:

            print("Loading video data")
            data = load_all_data("data/inference_model_train", num_splits=1,
                        display=self.display, no_transitions=False)
            xtrain, xval, _, ytrain, yval, _ = data
            self.train_data = xtrain
            self.val_data = xval
            data = load_all_data("data/inference_model_test", num_splits=0,
                        display=self.display, no_transitions=False)  
            xtest, _, _, ytest, _, _ = data
            self.test_data = xtest

            self.img_model, self.categorical = get_pred_model(self.name)

        x = self.get_data_by_type(typ)
        # make predictions
        print("Making", typ, "imagemodel predictions")
        preds = self.img_model.predict(x, verbose=1)
        print("Saving", savepath)
        np.savez(savepath, preds=preds, categorical=self.categorical)
        return preds


def load_predictions(name, display=False, repredict=False):
    """
    returns:
        xpredtrain, xpredval, xpredtest, categorical (bool)
    """
    # load predictions, or run predictions if there is unpredicted data
    loader = PredLoader(name, display, repredict=repredict)

    xpredtrain, c1 = loader.load("train")
    xpredval, c2 = loader.load("val")
    xpredtest, c3 = loader.load("test")

    assert c1 == c2 == c3

    return xpredtrain, xpredval, xpredtest, c1


if __name__ == "__main__":
    # just for testing
    parser = argparse.ArgumentParser()
    parser.add_argument("--name",required=True)
    parser.add_argument("--repredict",action="store_true")
    args = parser.parse_args()
    load_predictions(args.name, repredict=args.repredict)

