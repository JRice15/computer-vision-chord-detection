import keras
import numpy as np
import tensorflow as tf
from keras import backend as K
from keras import layers
from keras.layers import (Activation, Add, BatchNormalization, Concatenate,
                          Conv2D, Dense, Dropout, Flatten,
                          GlobalAveragePooling2D, Input, Lambda, MaxPooling2D,
                          Multiply, ReLU, Reshape, Softmax, Bidirectional, LSTM)
from keras.models import Model
import logging
from keras.optimizers import Adam


INPUT_LEN = 64


def group_sequences(x, y, step=10):
    """
    split data into input_length sized pieces
    args:
        step: step size between sequences
    """
    X = np.empty((len(x)//step, INPUT_LEN)+x.shape[1:], dtype=x.dtype)
    Y = np.empty((len(y)//step, INPUT_LEN)+y.shape[1:], dtype=y.dtype)
    for batch_ind,ind in enumerate(range(INPUT_LEN, len(x), step)):
        X[batch_ind] = x[ind-INPUT_LEN:ind]
        Y[batch_ind] = y[ind-INPUT_LEN:ind]
    return X, Y


class MyConv1DTranspose(tf.keras.layers.Layer):
    """tf forgot about this one, until v2.3 I guess"""

    def __init__(self, filters, kernel_size, strides=1, padding='valid', **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.conv2dtranspose = tf.keras.layers.Conv2DTranspose(
          filters, (kernel_size, 1), (strides, 1), padding
        )

    def call(self, x):
        x = tf.expand_dims(x, axis=2)
        x = self.conv2dtranspose(x)
        x = tf.squeeze(x, axis=2)
        return x
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "filters": self.filters,
            "kernel_size": self.kernel_size,
            "strides": self.strides,
            "padding": self.padding
        })
        return config



def inference_autoencoder(x, depth, target_depth):
    if depth >= 128:
        depth2 = depth3 = int(1.5 * depth)
    else:
        depth2 = 2 * depth
        depth3 = 3 * depth

    ### Encoding (8x downsample)
    x = layers.Conv1D(depth, kernel_size=5, strides=2, use_bias=False, 
            padding="same")(x)
    # x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.SpatialDropout1D(0.2)(x)

    x = layers.Conv1D(depth2, kernel_size=5, strides=2, use_bias=False, 
            padding="same")(x)
    # x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.SpatialDropout1D(0.2)(x)

    x = layers.Conv1D(depth3, kernel_size=5, strides=2, use_bias=False, 
            padding="same")(x)
    # x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.SpatialDropout1D(0.4)(x)

    ### Middle
    x = layers.Conv1D(depth3, kernel_size=5, strides=1, use_bias=False, 
            padding="same")(x)
    # x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.SpatialDropout1D(0.4)(x)
    x = layers.LSTM(depth3, return_sequences=True, 
            # go_backwards=True, 
            dropout=0.4, 
            name="lstm1")(x)
    # x = layers.BatchNormalization()(x)
    # x = layers.LSTM(depth3, return_sequences=True,
    #         # go_backwards=True, 
    #         # dropout=0.4, 
    #         name="lstm2")(x)
    # x = layers.BatchNormalization()(x)

    ### Decoding
    # upsample 2x with convolution
    x = MyConv1DTranspose(depth2, kernel_size=5, strides=2, padding="same")(x)
    # get to proper depth
    while x.shape[-1] > target_depth:
        new_depth = max(x.shape[-1] // 2, target_depth)
        # x = BatchNormalization()(x)
        x = layers.SpatialDropout1D(0.4)(x)
        x = layers.ReLU()(x)
        x = Dense(new_depth)(x)
    x = ReLU()(x)
    # x = layers.SpatialDropout1D(0.4)(x)
    x = Dense(target_depth)(x)

    # we don't have to have super fine-grain precision. missing a 
    #  transition by 4 frames is fine
    x = layers.UpSampling1D(4)(x)

    return x


def v2_simple(x, depth, target_depth):
    depth2 = max(depth//2, target_depth)
    depth3 = max(depth2//2, target_depth)

    x = layers.SpatialDropout1D(0.6)(x)

    x = Bidirectional(LSTM(depth2, return_sequences=True), merge_mode='concat')(x)
    x = layers.SpatialDropout1D(0.4)(x)

    x = layers.Conv1D(depth3, kernel_size=5, padding="same")(x)
    x = ReLU()(x)
    x = layers.SpatialDropout1D(0.4)(x)

    x = layers.Conv1D(target_depth, kernel_size=1)(x)

    return x

def v3_sepconv(x, depth, target_depth):
    # x = layers.SpatialDropout1D(0.6)(x)
    # x = Bidirectional(LSTM(depth, return_sequences=True), merge_mode='concat')(x)

    x = layers.SeparableConv1D(2*depth, kernel_size=3, strides=1, padding="same")(x)
    x = ReLU()(x)
    x = BatchNormalization()(x)
    # x = layers.SpatialDropout1D(0.4)(x)

    x = layers.SeparableConv1D(2*depth, kernel_size=3, strides=1, padding="same")(x)
    x = ReLU()(x)
    # x = layers.SpatialDropout1D(0.4)(x)
    x = BatchNormalization()(x)

    x = layers.SeparableConv1D(2*depth, kernel_size=3, strides=1, padding="same")(x)
    x = ReLU()(x)
    # x = BatchNormalization()(x)
    x = layers.SpatialDropout1D(0.6)(x)

    x = layers.SeparableConv1D(2*depth, kernel_size=3, strides=1, padding="same")(x)
    x = ReLU()(x)
    # x = BatchNormalization()(x)
    x = layers.SpatialDropout1D(0.6)(x)

    x = layers.SeparableConv1D(2*depth, kernel_size=3, strides=1, padding="same")(x)
    x = ReLU()(x)
    # x = BatchNormalization()(x)
    x = layers.SpatialDropout1D(0.6)(x)

    x = layers.Conv1D(target_depth, kernel_size=1, strides=1, padding="same")(x)
    # x = MyConv1DTranspose(target_depth, kernel_size=1, strides=8, padding="same")(x)
    # x = layers.UpSampling1D(4)(x)
    return x

def v4_simple(x, depth, target_depth):
    depth2 = max(depth//2, target_depth)
    depth3 = max(depth2//2, target_depth)

    x = BatchNormalization()(x)
    x = layers.SpatialDropout1D(0.6)(x)

    x = Bidirectional(LSTM(depth, return_sequences=True), merge_mode='concat')(x)
    x = layers.SpatialDropout1D(0.4)(x)

    x = layers.SeparableConv1D(depth, kernel_size=5, padding="same")(x)
    x = ReLU()(x)
    x = layers.SpatialDropout1D(0.4)(x)

    x = layers.Conv1D(target_depth, kernel_size=1)(x)

    return x

def make_inference_model(inpt_shape, output_shape, categorical=True):
    """
    1d convolutional autoencoder
    returns:
        model
    """
    inpt = layers.Input(inpt_shape)
    x = inpt

    if len(inpt_shape) > 2:
        inpt_length, numstrings, numfrets = inpt_shape
        depth = numstrings * numfrets
        # flatten categorical input
        x = layers.Reshape((inpt_length, depth), name="reshape1")(x)
    else:
        inpt_length, depth = inpt_shape
    if len(output_shape) > 2:
        inpt_length, numstrings, numfrets = output_shape
        target_depth = numstrings * numfrets
    else:
        inpt_length, target_depth = inpt_shape
    
    # x = inference_autoencoder(x, depth, target_depth)
    # x = v2_simple(x, depth, target_depth)
    # x = v3_sepconv(x, depth, target_depth)
    x = v4_simple(x, depth, target_depth)

    if x.shape != output_shape:
        x = layers.Reshape(output_shape, name="reshape2")(x)

    if categorical:
        x = layers.Softmax(axis=-1)(x)
    else:
        x = layers.Activation('sigmoid')(x)
        x = layers.Lambda(lambda v: (numfrets-1) * v)(x)

    model = Model(inpt, x)

    # regularize adjacent timesteps to be similar
    # REG_WEIGHT = 1.0
    # diff = K.abs(x[:-1] - x[1:])
    # diff_loss = K.mean(diff) * REG_WEIGHT
    # model.add_loss(diff_loss)
    # model.add_metric(diff_loss, aggregation='mean', name='diff_loss')

    return model

