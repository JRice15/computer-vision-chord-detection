import keras
import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.layers import (Activation, Add, BatchNormalization, Concatenate,
                          Conv2D, Dense, Flatten, GlobalAveragePooling2D,
                          Input, MaxPooling2D, ReLU, Multiply, Lambda, Dropout, Reshape)
from keras.models import Model
from keras.optimizers import Adam

if not tf.__version__.startswith("2.2") or not keras.__version__.startswith("2.4.3"):
    print("This code was written with TensorFlow 2.2 and Keras 2.4.3, and may fail on your version:")
print("tf:", tf.__version__)
print("keras:", keras.__version__)

MAX_FRET = 6

def xception(input_shape):
    """
    keras xception network. see https://keras.io/api/applications/
    """
    inpt = Input(input_shape)
    base = keras.applications.Xception(include_top=False, weights=None, 
                input_shape=input_shape, pooling='avg')
    x = base(inpt)

    x = Dense(256)(x)
    x = ReLU()(x)
    x = Dropout(0.4)(x)

    x = Dense(32)(x)
    x = ReLU()(x)
    x = Dropout(0.4)(x)

    x = Dense(6)(x)
    # sigmoid limits to between 0 and 1. Then we multiply by the constant to
    # allow a range from 0 to MAX_FRET to be predicted for each string
    x = Activation('sigmoid')(x)
    x = Lambda(lambda v: MAX_FRET * v)(x)

    return Model(inpt, x)

def mobilenetv2(input_shape):
    """
    keras mobilenetv2
    """
    inpt = Input(input_shape)
    base = keras.applications.MobileNetV2(include_top=False, weights=None,
                input_shape=input_shape, alpha=1.0)
    # x = keras.applications.mobilenet_v2.preprocess_input(inpt)
    x = base(inpt)
    # x = GlobalAveragePooling2D()(x)

    x = Dense(128)(x)
    x = ReLU()(x)
    x = Dropout(0.5)(x)

    x = Dense(8)(x)
    x = ReLU()(x)
    x = Dropout(0.5)(x)

    x = Flatten()(x)
    x = Dense(64)(x)
    x = ReLU()(x)
    x = Dropout(0.5)(x)

    x = Dense(32)(x)
    x = ReLU()(x)
    x = Dropout(0.5)(x)

    x = Dense(6)(x)
    # sigmoid limits to between 0 and 1. Then we multiply by the constant to
    # allow a range from 0 to MAX_FRET to be predicted for each string
    x = Activation('sigmoid')(x)
    x = Lambda(lambda v: MAX_FRET * v)(x)

    return Model(inpt, x)


def make_model(name, input_shape):
    """
    get model from case insensitive name
    """
    name = name.lower()
    if name == "xception":
        return xception(input_shape)
    if name == "mobilenetv2":
        return mobilenetv2(input_shape)
    else:
        raise ValueError("no model named '" + name + "'")





