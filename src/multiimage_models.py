import logging

import keras
import numpy as np
import tensorflow as tf
from keras import backend as K
from keras import layers
from keras.applications import imagenet_utils
from keras.layers import (LSTM, Activation, Add, BatchNormalization,
                          Bidirectional, Concatenate, Conv1D, Conv2D, Dense,
                          Dropout, Flatten, GlobalAveragePooling2D, Input,
                          Lambda, MaxPooling2D, Multiply, ReLU, Reshape,
                          Softmax)
from keras.models import Model
from keras.optimizers import Adam

from src.cv_helpers import *
from src.image_models import fret_accuracy, get_output_shape

if not tf.__version__.startswith("2.2") or not keras.__version__.startswith("2.4.3"):
    print("This code was written with TensorFlow 2.2 and Keras 2.4.3, and may fail on your version:")
print("tf:", tf.__version__)
print("keras:", keras.__version__)

MAX_FRET = 5

def group_image_sequences(x, y, num_inputs, step=10):
    """
    split data into input_length sized pieces
    args:
        num_inputs: length of each sequence
        step: step size between sequences
    """
    X = np.empty((len(x)//step, num_inputs)+x.shape[1:], dtype=x.dtype)
    Y = np.empty((len(y)//step, num_inputs)+y.shape[1:], dtype=x.dtype)
    for batch_ind,ind in enumerate(range(num_inputs, len(x), step)):
        X[batch_ind] = x[ind-num_inputs:ind]
        Y[batch_ind] = y[ind-num_inputs:ind]
        # showim(X[batch_ind][0])
    return X, Y


def make_multiimage_model(name, num_inputs, img_shape, output_confidences):
    """
    get imagemodel from case insensitive name
    args:
        model name
        img shape
        output_confidences (bool): whether loss to output a prediction, or a 
            softmax confidence for each string
    """
    inpt = Input((num_inputs,)+img_shape)
    # unpack sequence into individual images
    inputs = []
    for i in range(num_inputs):
        inputs.append(inpt[:,i])

    name = name.lower()
    if name == "xception":
        base = keras.applications.Xception(include_top=False, weights=None, 
                input_shape=img_shape)
    elif name == "mobilenetv2":
        base = keras.applications.MobileNetV2(include_top=False, weights=None,
                input_shape=img_shape, alpha=1.0)
    else:
        raise ValueError("no model named '" + name + "'")

    xs = []
    for x in inputs:
        x = base(x)
        x = GlobalAveragePooling2D()(x)
        xs.append(x)

    # recombine
    xs = [Reshape((1,)+x.shape[1:])(x) for x in xs]
    x = Concatenate(axis=1)(xs)
    #x = layers.SpatialDropout1D(0.6)(x)

    # process together
    x = Bidirectional(LSTM(128, return_sequences=True), merge_mode='concat')(x)
    x = Bidirectional(LSTM(64, return_sequences=True), merge_mode='sum')(x)
    #x = layers.SpatialDropout1D(0.4)(x)

    #x = layers.SeparableConv1D(64, kernel_size=8, padding="same")(x)
    #x = ReLU()(x)
    #x = layers.SpatialDropout1D(0.4)(x)

    if output_confidences:
        # if the maximum fret is 5, there are 6 options, because 0 is a possibility
        NUM_FRETS = MAX_FRET + 1
        x = Conv1D(6 * NUM_FRETS, kernel_size=1, padding="same")(x)
        x = Reshape((num_inputs, 6,NUM_FRETS))(x)
        x = Softmax(axis=-1)(x)

    else:
        x = Conv1D(6, kernel_size=1, padding="same")(x)
        # sigmoid limits to between 0 and 1. Then we multiply by the constant to
        # allow a range from 0 to MAX_FRET to be predicted for each string
        x = Activation('sigmoid')(x)
        x = Lambda(lambda v: MAX_FRET * v)(x)

    return Model(inpt, x)




